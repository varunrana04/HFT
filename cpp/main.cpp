#include "strategy_engine.h"
#include "delta_ws.h"
#include "delta_rest.h"
#include <iostream>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <future>
#include <atomic>
#include <csignal>

using namespace hft;

// ── Graceful shutdown ──────────────────────────────────────────────────────
static std::atomic<bool> g_running{true};
void handle_signal(int) { g_running = false; }

// ── Write JSON status to /var/www/html/status.json (served by NGINX) ───────
static void write_status(double bid, double ask, double alpha,
                         int64_t pos, double pnl) {
    // Write to tmp then rename for atomic update
    const char* path  = "/var/www/html/status.json";
    const char* tmp   = "/var/www/html/status.json.tmp";
    std::ofstream f(tmp);
    if (!f) {
        std::cerr << "[ERROR] Could not open " << tmp << " for writing! Permission denied?\n";
        return;
    }
    if (std::isnan(bid) || std::isinf(bid)) bid = 0.0;
    if (std::isnan(ask) || std::isinf(ask)) ask = 0.0;
    if (std::isnan(alpha) || std::isinf(alpha)) alpha = 0.0;
    if (std::isnan(pnl) || std::isinf(pnl)) pnl = 0.0;

    f << "{\"bid\":" << bid
      << ",\"ask\":" << ask
      << ",\"alpha\":" << alpha
      << ",\"pos\":" << pos
      << ",\"pnl\":" << pnl
      << ",\"ts\":" << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch()).count()
      << "}";
    f.close();
    std::rename(tmp, path);
}

int main() {
    std::signal(SIGINT,  handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::cout << "=======================================\n";
    std::cout << "🚀 Booting High-Frequency Trading Engine\n";
    std::cout << "=======================================\n";

    // 1. Load API keys
    const char* api_key_env    = std::getenv("DELTA_API_KEY");
    const char* api_secret_env = std::getenv("DELTA_API_SECRET");
    if (!api_key_env || !api_secret_env) {
        std::cerr << "[ERROR] Missing DELTA_API_KEY or DELTA_API_SECRET in environment.\n";
        return 1;
    }
    std::string api_key(api_key_env);
    std::string api_secret(api_secret_env);
    std::cout << "[INFO] Loaded Delta Exchange API Credentials.\n";

    // 2. Initialize Engine
    StrategyConfig cfg;
    cfg.min_warmup_ticks = 10;
    StrategyEngine engine(cfg);
    if (engine.load_lgbm_model("lgbm_signal_model.txt")) {
        std::cout << "[INFO] LightGBM Alpha Model loaded successfully.\n";
    } else {
        std::cerr << "[WARNING] Could not load lgbm_signal_model.txt! Using fallback weights.\n";
    }

    // 3. Initialize Gateways — Delta Exchange only
    const char* sym_env    = std::getenv("DELTA_SYMBOL");
    std::string delta_symbol = sym_env ? sym_env : "BTCUSD";
    std::cout << "[INFO] Trading symbol: " << delta_symbol << "\n";

    gateway::DeltaWs   ws_feed(delta_symbol);
    gateway::DeltaRest rest_client(api_key, api_secret);

    // 4. Connect Market Data WS
    std::cout << "[INFO] Connecting to Delta Exchange WebSocket...\n";
    ws_feed.start_live_feed(&engine);

    // 5. Initialize CSV Logger
    std::ofstream csv_log("market_data_log.csv", std::ios::app);
    if (csv_log.tellp() == 0)
        csv_log << "timestamp_ns,best_bid,best_ask,spread_bps,alpha,position,pnl\n";

    // 6. Write initial status so dashboard has data immediately
    write_status(0, 0, 0, 0, 0);
    std::cout << "[INFO] Dashboard status file: /var/www/html/status.json\n";
    std::cout << "[INFO] Engine is LIVE. Press Ctrl+C to stop.\n";

    // 7. Main loop — 10 Hz
    int ticks = 0;
    int64_t last_synthetic_trade_ms = 0;
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        ticks++;

        auto latest_book = engine.latest_book();
        
        // Inject a synthetic trade if the testnet is completely dead for > 1 second
        // This ensures the Avellaneda-Stoikov model continues to update quotes
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if (latest_book.timestamp_ns > 0 && (now_ms - last_synthetic_trade_ms > 1000)) {
            Trade dummy{};
            dummy.price = latest_book.mid_price();
            dummy.quantity = 0; // 0 volume doesn't mess up VPIN
            dummy.timestamp_ns = latest_book.timestamp_ns;
            dummy.side = Side::NONE;
            engine.on_trade(dummy, latest_book);
            last_synthetic_trade_ms = now_ms;
        }

        auto fv          = engine.last_features();
        double pnl       = engine.realized_pnl() + engine.unrealized_pnl();
        int64_t pos      = engine.position();

        double bid    = static_cast<double>(latest_book.best_bid_price) / 1e8;
        double ask    = static_cast<double>(latest_book.best_ask_price) / 1e8;
        double spread = (bid > 0) ? ((ask - bid) / bid * 10000.0) : 0.0;

        // Log to CSV
        if (latest_book.timestamp_ns > 0) {
            csv_log << latest_book.timestamp_ns << "," << bid << "," << ask << ","
                    << spread << "," << fv.combined_alpha << "," << pos << "," << pnl << "\n";
        }

        // Write dashboard status every tick (100ms)
        write_status(bid, ask, fv.combined_alpha, pos, pnl);

        // Order Execution Dispatcher
        if (engine.pending_order_.active && !engine.pending_order_.dispatched_to_rest) {
            Order o{};
            o.side     = engine.pending_order_.side;
            o.price    = engine.pending_order_.price;
            o.quantity = engine.pending_order_.qty;

            std::cout << "[EXEC] Dispatching "
                      << (o.side == Side::BID ? "BUY" : "SELL")
                      << " Qty " << static_cast<double>(o.quantity) / 1e8
                      << " @ "   << static_cast<double>(o.price)    / 1e8 << "\n";

            (void)std::async(std::launch::async, [&rest_client, o, delta_symbol]() {
                rest_client.cancel_all_orders(delta_symbol);
                rest_client.submit_order(o, delta_symbol);
            });

            engine.pending_order_.dispatched_to_rest = true;
        }

        // Heartbeat every 5 s
        if (ticks % 50 == 0) {
            std::cout << "[HEARTBEAT] Engine Running. Best Bid: " << bid
                      << " | Best Ask: " << ask
                      << " | Alpha: " << fv.combined_alpha << "\n";
        }
    }

    std::cout << "[INFO] Shutting down gracefully.\n";
    return 0;
}
