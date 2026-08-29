#include "strategy_engine.h"
#include "delta_ws.h"
#include "delta_rest.h"
#include <iostream>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <future>
#include <ixwebsocket/IXWebSocketServer.h>

using namespace hft;

int main() {
    std::cout << "=======================================\n";
    std::cout << "🚀 Booting High-Frequency Trading Engine\n";
    std::cout << "=======================================\n";

    // 1. Load API Keys
    const char* api_key_env = std::getenv("DELTA_API_KEY");
    const char* api_secret_env = std::getenv("DELTA_API_SECRET");

    if (!api_key_env || !api_secret_env) {
        std::cerr << "[ERROR] Missing DELTA_API_KEY or DELTA_API_SECRET in environment.\n";
        std::cerr << "Please set them before running the engine.\n";
        return 1;
    }

    std::string api_key(api_key_env);
    std::string api_secret(api_secret_env);

    std::cout << "[INFO] Loaded Delta Exchange API Credentials.\n";

    // 2. Initialize Engine
    StrategyEngine engine;
    
    // 3. Initialize Gateways — Delta Exchange only
    gateway::DeltaWs ws_feed("BTCUSD");          // Delta perpetual symbol
    gateway::DeltaRest rest_client(api_key, api_secret);

    // 4. Connect Market Data
    std::cout << "[INFO] Connecting to Delta Exchange WebSocket...\n";
    ws_feed.start_live_feed(&engine);

    // 5. Initialize CSV Logger
    std::ofstream csv_log("market_data_log.csv", std::ios::app);
    if (csv_log.tellp() == 0) {
        csv_log << "timestamp_ns,best_bid,best_ask,spread_bps,alpha,position,pnl\n";
    }

    // 6. Initialize Dashboard Server
    ix::WebSocketServer dashboard_server(8081, "0.0.0.0");
    dashboard_server.setOnClientMessageCallback(
        [](std::shared_ptr<ix::ConnectionState> connectionState, ix::WebSocket& webSocket, const ix::WebSocketMessagePtr& msg) {
            // Ignore incoming messages from dashboard clients
        }
    );
    bool res = dashboard_server.listenAndStart();
    if (!res) {
        std::cerr << "[ERROR] Failed to start Dashboard server." << std::endl;
    } else {
        std::cout << "[INFO] Dashboard WebSocket Server listening on ws://127.0.0.1:8081\n";
    }

    std::cout << "[INFO] Engine is LIVE. Press Ctrl+C to stop.\n";

    // Block main thread while websocket runs in background
    int ticks = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10Hz loop
        ticks++;
        
        auto latest_book = engine.latest_book();
        auto fv = engine.last_features();
        double pnl = engine.realized_pnl() + engine.unrealized_pnl();
        int64_t pos = engine.position();

        double bid = latest_book.best_bid_price / 1e8;
        double ask = latest_book.best_ask_price / 1e8;
        double spread = (bid > 0) ? ((ask - bid) / bid * 10000.0) : 0.0; // bps

        // Log to CSV
        if (latest_book.timestamp_ns > 0) {
            csv_log << latest_book.timestamp_ns << "," << bid << "," << ask << "," 
                    << spread << "," << fv.combined_alpha << "," << pos << "," << pnl << "\n";
        }

        // Send to dashboard clients
        std::ostringstream json;
        json << R"({"bid":)" << bid 
             << R"(,"ask":)" << ask
             << R"(,"alpha":)" << fv.combined_alpha
             << R"(,"pos":)" << pos
             << R"(,"pnl":)" << pnl << "}";
             
        std::string payload = json.str();
        for (auto client : dashboard_server.getClients()) {
            client->sendText(payload);
        }

        // Order Execution Dispatcher
        if (engine.pending_order_.active) {
            // Create a fake Order struct for the REST client
            Order o{};
            o.side = engine.pending_order_.side;
            o.price = engine.pending_order_.price;
            // Force 1 contract for Testnet safety if size is huge (or use computed qty if you prefer)
            o.quantity = engine.pending_order_.qty; 
            
            std::cout << "[EXEC] Dispatching " << (o.side == Side::BID ? "BUY" : "SELL")
                      << " Order: Qty " << o.quantity / 1e8 << " @ " << o.price / 1e8 << "\n";

            // Fire and forget via async; cast to void to suppress nodiscard warning
            (void)std::async(std::launch::async, [&rest_client, o]() {
                rest_client.submit_order(o, "BTCUSD");
            });

            // Mark as dispatched
            engine.pending_order_.active = false;
        }

        // Print heartbeat occasionally (every 5 seconds = 50 ticks)
        if (ticks % 50 == 0) {
            std::cout << "[HEARTBEAT] Engine Running. Best Bid: " << bid 
                      << " | Best Ask: " << ask << " | Alpha: " << fv.combined_alpha << std::endl;
        }
    }

    return 0;
}
