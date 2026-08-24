#include "binance_ws.h"
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

struct BinanceWs::Impl {
    simdjson::ondemand::parser parser;
    ix::WebSocket webSocket;
    std::string ws_url = "wss://fstream.binance.com";
    std::string ws_path;
    
    Impl(const std::string& symbol) {
        // Connect to the base websocket endpoint, we will subscribe dynamically on Open
        ws_path = "/ws";
    }
};

BinanceWs::BinanceWs(const std::string& symbol) 
    : symbol_(symbol), impl_(std::make_unique<Impl>(symbol)) {
    ix::initNetSystem();
}

BinanceWs::~BinanceWs() {
    stop();
    ix::uninitNetSystem();
}

bool BinanceWs::initialize() {
    return true;
}

void BinanceWs::poll_loop(
    std::function<void(const Trade&)> on_trade_callback,
    std::function<void(const BookSnapshot&)> on_book_callback
) {
    on_trade_callback_ = std::move(on_trade_callback);
    on_book_callback_ = std::move(on_book_callback);
}

void BinanceWs::start_live_feed(hft::StrategyEngine* engine) {
    engine_ = engine;
    
    std::string url = impl_->ws_url + impl_->ws_path;
    impl_->webSocket.setUrl(url);
    
    // Let IXWebSocket use the default system CA certificates (now installed via Docker)
    ix::SocketTLSOptions tlsOptions;
    impl_->webSocket.setTLSOptions(tlsOptions);
    
    // Binance requires ping/pong to keep connection alive
    impl_->webSocket.setPingInterval(30);
    
    impl_->webSocket.setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Message) {
            try {
                simdjson::ondemand::document doc;
                auto error = impl_->parser.iterate(msg->str).get(doc);
                if (error) return;

                simdjson::ondemand::object obj = doc.get_object();
                std::string_view stream_name;
                if (obj["stream"].get(stream_name) == simdjson::SUCCESS) {
                    simdjson::ondemand::object data = obj["data"].get_object();
                    
                    std::string_view event_type;
                    if (data["e"].get(event_type) == simdjson::SUCCESS) {
                        
                        if (event_type == "trade") {
                            Trade t{};
                            std::string_view p_str;
                            std::string_view q_str;
                            bool is_buyer_maker = false;
                            
                            data["p"].get(p_str);
                            data["q"].get(q_str);
                            data["m"].get(is_buyer_maker);
                            
                            t.price = std::stod(std::string(p_str)) * 1e8;
                            t.quantity = std::stod(std::string(q_str)) * 1e8;
                            t.side = is_buyer_maker ? Side::ASK : Side::BID;
                            t.timestamp_ns = std::chrono::system_clock::now().time_since_epoch().count();
                            t.quality = DataQuality::VALID;
                            
                            if (engine_) {
                                engine_->on_trade(t, latest_book_);
                            } else if (on_trade_callback_) {
                                on_trade_callback_(t);
                            }
                        }
                        else if (event_type == "depthUpdate") {
                            BookSnapshot book{};
                            book.timestamp_ns = std::chrono::system_clock::now().time_since_epoch().count();
                            book.quality = DataQuality::VALID;
                            
                            // Bids
                            int bid_idx = 0;
                            for (auto bid : data["b"]) {
                                if (bid_idx >= 5) break;
                                auto arr = bid.get_array();
                                auto it = arr.begin();
                                std::string_view p, q;
                                (*it).get(p); ++it;
                                (*it).get(q);
                                book.bids[bid_idx].price = std::stod(std::string(p)) * 1e8;
                                book.bids[bid_idx].quantity = std::stod(std::string(q)) * 1e8;
                                bid_idx++;
                            }
                            book.bid_count = bid_idx;

                            // Asks
                            int ask_idx = 0;
                            for (auto ask : data["a"]) {
                                if (ask_idx >= 5) break;
                                auto arr = ask.get_array();
                                auto it = arr.begin();
                                std::string_view p, q;
                                (*it).get(p); ++it;
                                (*it).get(q);
                                book.asks[ask_idx].price = std::stod(std::string(p)) * 1e8;
                                book.asks[ask_idx].quantity = std::stod(std::string(q)) * 1e8;
                                ask_idx++;
                            }
                            book.ask_count = ask_idx;
                            
                            if (bid_idx > 0) {
                                book.best_bid_price = book.bids[0].price;
                                book.best_bid_qty = book.bids[0].quantity;
                            } else {
                                book.best_bid_price = INVALID_PRICE;
                            }

                            if (ask_idx > 0) {
                                book.best_ask_price = book.asks[0].price;
                                book.best_ask_qty = book.asks[0].quantity;
                            } else {
                                book.best_ask_price = INVALID_PRICE;
                            }
                            
                            latest_book_ = book; // Cache for on_trade
                            
                            if (engine_) {
                                engine_->on_book_update(book);
                            } else if (on_book_callback_) {
                                on_book_callback_(book);
                            }
                        }
                    }
                }
            } catch (...) {
                // Ignore parsing errors for speed
            }
        }
        else if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "[BinanceWs] Connected to Binance! Sending subscribe request..." << std::endl;
            // Dynamically subscribe to streams to avoid URL parsing bugs in IXWebSocket
            std::string lower_symbol = symbol_;
            for(auto& c : lower_symbol) c = std::tolower(c);
            
            std::string payload = R"({"method": "SUBSCRIBE", "params": [")" + lower_symbol + R"(@trade", ")" + lower_symbol + R"(@depth5@100ms"], "id": 1})";
            impl_->webSocket.send(payload);
        }
        else if (msg->type == ix::WebSocketMessageType::Close) {
            std::cout << "[BinanceWs] Disconnected from Binance!" << std::endl;
        }
        else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cout << "[BinanceWs] Connection Error: " << msg->errorInfo.reason << std::endl;
        }
    });

    impl_->webSocket.start();
    running_ = true;
}

void BinanceWs::stop() {
    running_ = false;
    impl_->webSocket.stop();
}

} // namespace gateway
} // namespace hft
