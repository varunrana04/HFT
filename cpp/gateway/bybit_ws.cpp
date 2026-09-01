#include "bybit_ws.h"
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

struct BybitWs::Impl {
    simdjson::dom::parser parser;
    ix::WebSocket webSocket;
    std::string ws_url;
    
    Impl(const std::string& symbol) {
        (void)symbol;
        const char* env_url = std::getenv("BYBIT_WS_BASE_URL");
        ws_url = env_url ? env_url : "wss://stream.bybit.com/v5/public/linear";
    }
};

BybitWs::BybitWs(const std::string& symbol) 
    : symbol_(symbol), impl_(std::make_unique<Impl>(symbol)) {
    ix::initNetSystem();
    latest_book_.best_bid_price = INVALID_PRICE;
    latest_book_.best_ask_price = INVALID_PRICE;
}

BybitWs::~BybitWs() {
    stop();
    ix::uninitNetSystem();
}

bool BybitWs::initialize() {
    return true;
}

void BybitWs::poll_loop(
    std::function<void(const Trade&)> on_trade_callback,
    std::function<void(const BookSnapshot&)> on_book_callback
) {
    on_trade_callback_ = std::move(on_trade_callback);
    on_book_callback_ = std::move(on_book_callback);
}

void BybitWs::start_live_feed(hft::StrategyEngine* engine) {
    engine_ = engine;
    
    impl_->webSocket.setUrl(impl_->ws_url);
    
    ix::SocketTLSOptions tlsOptions;
    impl_->webSocket.setTLSOptions(tlsOptions);
    impl_->webSocket.setPingInterval(20);
    impl_->webSocket.enableAutomaticReconnection();
    impl_->webSocket.setMaxWaitBetweenReconnectionRetries(5000); 
    impl_->webSocket.disablePerMessageDeflate();
    
    ix::WebSocketHttpHeaders headers;
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";
    impl_->webSocket.setExtraHeaders(headers);
    
    std::string symbol_upper = symbol_;
    impl_->webSocket.setOnMessageCallback([this, symbol_upper](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "[BybitWs] Connected to " << impl_->ws_url << std::endl;
            
            // Send subscription message
            std::string sub_msg = "{\"op\": \"subscribe\", \"args\": [\"orderbook.50." + symbol_upper + "\", \"publicTrade." + symbol_upper + "\"]}";
            impl_->webSocket.send(sub_msg);
        } else if (msg->type == ix::WebSocketMessageType::Message) {
            try {
                simdjson::dom::element doc;
                auto error = impl_->parser.parse(msg->str).get(doc);
                if (error) return;

                simdjson::dom::object obj;
                if (doc.get_object().get(obj) != simdjson::SUCCESS) return;
                
                std::string_view topic;
                if (obj["topic"].get(topic) != simdjson::SUCCESS) return;

                if (topic.find("publicTrade") != std::string_view::npos) {
                    simdjson::dom::array data_arr;
                    if (obj["data"].get(data_arr) == simdjson::SUCCESS) {
                        for (simdjson::dom::element item : data_arr) {
                            simdjson::dom::object trade_obj;
                            if (item.get_object().get(trade_obj) != simdjson::SUCCESS) continue;
                            
                            Trade trade{};
                            int64_t ts = 0;
                            if (trade_obj["T"].get(ts) == simdjson::SUCCESS) {
                                trade.timestamp_ns = ts * 1000000;
                            }
                            
                            std::string_view price_str;
                            if (trade_obj["p"].get(price_str) == simdjson::SUCCESS) {
                                trade.price = static_cast<int64_t>(std::stod(std::string(price_str)) * 1e8);
                            }
                            
                            std::string_view qty_str;
                            if (trade_obj["v"].get(qty_str) == simdjson::SUCCESS) {
                                trade.quantity = static_cast<int64_t>(std::stod(std::string(qty_str)) * 1e8);
                            }
                            
                            std::string_view side_str;
                            if (trade_obj["S"].get(side_str) == simdjson::SUCCESS) {
                                trade.side = (side_str == "Buy") ? Side::BID : Side::ASK;
                            }
                            
                            if (engine_) {
                                engine_->on_trade(trade, latest_book_);
                            } else if (on_trade_callback_) {
                                on_trade_callback_(trade);
                            }
                        }
                    }
                } else if (topic.find("orderbook") != std::string_view::npos) {
                    simdjson::dom::object data_obj;
                    if (obj["data"].get(data_obj) == simdjson::SUCCESS) {
                        simdjson::dom::array bids;
                        simdjson::dom::array asks;
                        
                        bool has_bids = (data_obj["b"].get(bids) == simdjson::SUCCESS);
                        bool has_asks = (data_obj["a"].get(asks) == simdjson::SUCCESS);
                        
                        if (!has_bids && !has_asks) return;
                        
                        latest_book_.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                            
                        if (has_bids && bids.size() > 0) {
                            simdjson::dom::array first_bid;
                            if (bids.at(0).get(first_bid) == simdjson::SUCCESS && first_bid.size() >= 2) {
                                std::string_view price_str;
                                std::string_view qty_str;
                                if (first_bid.at(0).get(price_str) == simdjson::SUCCESS && first_bid.at(1).get(qty_str) == simdjson::SUCCESS) {
                                    latest_book_.best_bid_price = static_cast<int64_t>(std::stod(std::string(price_str)) * 1e8);
                                    latest_book_.best_bid_qty = static_cast<int64_t>(std::stod(std::string(qty_str)) * 1e8);
                                    latest_book_.bid_count = static_cast<int32_t>(bids.size());
                                }
                            }
                        }
                        
                        if (has_asks && asks.size() > 0) {
                            simdjson::dom::array first_ask;
                            if (asks.at(0).get(first_ask) == simdjson::SUCCESS && first_ask.size() >= 2) {
                                std::string_view price_str;
                                std::string_view qty_str;
                                if (first_ask.at(0).get(price_str) == simdjson::SUCCESS && first_ask.at(1).get(qty_str) == simdjson::SUCCESS) {
                                    latest_book_.best_ask_price = static_cast<int64_t>(std::stod(std::string(price_str)) * 1e8);
                                    latest_book_.best_ask_qty = static_cast<int64_t>(std::stod(std::string(qty_str)) * 1e8);
                                    latest_book_.ask_count = static_cast<int32_t>(asks.size());
                                }
                            }
                        }
                        
                        if (engine_ && latest_book_.is_valid()) {
                            engine_->on_book_update(latest_book_);
                        } else if (on_book_callback_ && latest_book_.is_valid()) {
                            on_book_callback_(latest_book_);
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[BybitWs] JSON parse error: " << e.what() << "\n";
            }
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cerr << "[BybitWs] Connection error: " << msg->errorInfo.reason << std::endl;
        }
    });
    
    impl_->webSocket.start();
    running_ = true;
}

void BybitWs::stop() {
    if (running_) {
        impl_->webSocket.stop();
        running_ = false;
    }
}

} // namespace gateway
} // namespace hft
