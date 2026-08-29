#include "delta_ws.h"
#include "strategy_engine.h"
#include <iostream>
#include <chrono>

// Using simdjson single header
#include <simdjson/simdjson.h>

// IXWebSocket
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>

namespace hft {
namespace gateway {

struct DeltaWs::Impl {
    simdjson::dom::parser parser;
    ix::WebSocket webSocket;
    // Read WS URL from env, fallback to global production endpoint
    std::string ws_url;
    
    Impl(const std::string& symbol) {
        (void)symbol;
        const char* env_url = std::getenv("DELTA_WS_URL");
        ws_url = env_url ? env_url : "wss://socket.delta.exchange";
    }
};

DeltaWs::DeltaWs(const std::string& symbol) 
    : symbol_(symbol), impl_(std::make_unique<Impl>(symbol)) {
    ix::initNetSystem();
    latest_book_.best_bid_price = INVALID_PRICE;
    latest_book_.best_ask_price = INVALID_PRICE;
}

DeltaWs::~DeltaWs() {
    stop();
    ix::uninitNetSystem();
}

bool DeltaWs::initialize() {
    return true;
}

void DeltaWs::poll_loop(
    std::function<void(const Trade&)> on_trade_callback,
    std::function<void(const BookSnapshot&)> on_book_callback
) {
    on_trade_callback_ = std::move(on_trade_callback);
    on_book_callback_ = std::move(on_book_callback);
}

void DeltaWs::start_live_feed(hft::StrategyEngine* engine) {
    engine_ = engine;
    
    impl_->webSocket.setUrl(impl_->ws_url);
    
    ix::SocketTLSOptions tlsOptions;
    impl_->webSocket.setTLSOptions(tlsOptions);
    impl_->webSocket.setPingInterval(30);
    impl_->webSocket.enableAutomaticReconnection();
    impl_->webSocket.setMaxWaitBetweenReconnectionRetries(5000); 
    impl_->webSocket.disablePerMessageDeflate();
    
    ix::WebSocketHttpHeaders headers;
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";
    impl_->webSocket.setExtraHeaders(headers);
    
    impl_->webSocket.setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Message) {
            try {
                simdjson::dom::element doc;
                auto error = impl_->parser.parse(msg->str).get(doc);
                if (error) return;

                simdjson::dom::object obj;
                if (doc.get_object().get(obj) != simdjson::SUCCESS) return;
                
                // Check if it's an orderbook message
                bool is_book = false;
                simdjson::dom::array buy_arr;
                if (obj["buy"].get(buy_arr) == simdjson::SUCCESS) {
                    is_book = true;
                }

                // Check if it's a trade message
                bool is_trade = false;
                simdjson::dom::array trades_arr;
                if (obj["trades"].get(trades_arr) == simdjson::SUCCESS) {
                    is_trade = true;
                }

                if (is_trade) {
                    for (auto trade_val : trades_arr) {
                        simdjson::dom::object trade_obj;
                        if (trade_val.get(trade_obj) != simdjson::SUCCESS) continue;
                        
                        Trade t{};
                        std::string_view buyer_role;
                        
                        double price = 0.0;
                        double size = 0.0;
                        (void)trade_obj["price"].get(price);
                        (void)trade_obj["size"].get(size);
                        (void)trade_obj["buyer_role"].get(buyer_role);
                        
                        t.price = static_cast<int64_t>(price * 1e8);
                        t.quantity = static_cast<int64_t>(size * 1e8); 
                        t.side = (buyer_role == "taker") ? Side::BID : Side::ASK;
                        
                        uint64_t ts = 0;
                        if (trade_obj["timestamp"].get(ts) == simdjson::SUCCESS) {
                            t.timestamp_ns = ts * 1000; // microseconds to ns
                        } else {
                            t.timestamp_ns = std::chrono::system_clock::now().time_since_epoch().count();
                        }
                        
                        t.quality = DataQuality::VALID;
                        
                        if (engine_) {
                            engine_->on_trade(t, latest_book_);
                        } else if (on_trade_callback_) {
                            on_trade_callback_(t);
                        }
                    }
                }
                
                if (is_book) {
                    uint64_t ts = 0;
                    if (obj["timestamp"].get(ts) == simdjson::SUCCESS) {
                        latest_book_.timestamp_ns = ts * 1000; // microseconds to ns
                    } else {
                        latest_book_.timestamp_ns = std::chrono::system_clock::now().time_since_epoch().count();
                    }
                    latest_book_.quality = DataQuality::VALID;
                    
                    // Bids (buy)
                    if (buy_arr.size() > 0) {
                        int bid_idx = 0;
                        for (auto bid : buy_arr) {
                            if (bid_idx >= 5) break; 
                            simdjson::dom::object bid_obj;
                            if (bid.get(bid_obj) != simdjson::SUCCESS) continue;
                            
                            std::string_view p;
                            double q = 0.0;
                            (void)bid_obj["limit_price"].get(p);
                            (void)bid_obj["size"].get(q);
                            
                            latest_book_.bids[bid_idx].price = static_cast<int64_t>(std::stod(std::string(p)) * 1e8);
                            latest_book_.bids[bid_idx].quantity = static_cast<int64_t>(q * 1e8);
                            bid_idx++;
                        }
                        latest_book_.bid_count = bid_idx;
                        if (bid_idx > 0) {
                            latest_book_.best_bid_price = latest_book_.bids[0].price;
                            latest_book_.best_bid_qty = latest_book_.bids[0].quantity;
                        } else {
                            latest_book_.best_bid_price = INVALID_PRICE;
                        }
                    }

                    // Asks (sell)
                    simdjson::dom::array sell_arr;
                    if (obj["sell"].get(sell_arr) == simdjson::SUCCESS) {
                        int ask_idx = 0;
                        for (auto ask : sell_arr) {
                            if (ask_idx >= 5) break;
                            simdjson::dom::object ask_obj;
                            if (ask.get(ask_obj) != simdjson::SUCCESS) continue;
                            
                            std::string_view p;
                            double q = 0.0;
                            (void)ask_obj["limit_price"].get(p);
                            (void)ask_obj["size"].get(q);
                            
                            latest_book_.asks[ask_idx].price = static_cast<int64_t>(std::stod(std::string(p)) * 1e8);
                            latest_book_.asks[ask_idx].quantity = static_cast<int64_t>(q * 1e8);
                            ask_idx++;
                        }
                        latest_book_.ask_count = ask_idx;
                        if (ask_idx > 0) {
                            latest_book_.best_ask_price = latest_book_.asks[0].price;
                            latest_book_.best_ask_qty = latest_book_.asks[0].quantity;
                        } else {
                            latest_book_.best_ask_price = INVALID_PRICE;
                        }
                    }
                    
                    if (engine_) {
                        engine_->on_book_update(latest_book_);
                    } else if (on_book_callback_) {
                        on_book_callback_(latest_book_);
                    }
                }
            } catch (...) {
                // Ignore parsing errors for speed
            }
        }
        else if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "[DeltaWs] Connected to Delta Exchange! Sending subscribe request..." << std::endl;
            
            std::string payload = R"({
                "type": "subscribe",
                "payload": {
                    "channels": [
                        { "name": "l2_orderbook", "symbols": [")" + symbol_ + R"("] },
                        { "name": "all_trades", "symbols": [")" + symbol_ + R"("] }
                    ]
                }
            })";
            impl_->webSocket.send(payload);
        }
        else if (msg->type == ix::WebSocketMessageType::Close) {
            std::cout << "[DeltaWs] Disconnected from Delta Exchange! Code: " << msg->closeInfo.code << ", Reason: " << msg->closeInfo.reason << ". Auto-reconnecting..." << std::endl;
        }
        else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cout << "[DeltaWs] Connection Error: " << msg->errorInfo.reason << ". Retries: " << msg->errorInfo.retries << std::endl;
        }
    });

    impl_->webSocket.start();
    running_ = true;
}

void DeltaWs::stop() {
    running_ = false;
    impl_->webSocket.stop();
}

} // namespace gateway
} // namespace hft
