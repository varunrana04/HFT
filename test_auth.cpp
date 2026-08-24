#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    ix::initNetSystem();
    ix::WebSocket webSocket;
    webSocket.setUrl("wss://fstream-auth.binance.com/ws");
    
    webSocket.disablePerMessageDeflate();
    
    ix::WebSocketHttpHeaders headers;
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";
    webSocket.setExtraHeaders(headers);
    
    webSocket.setOnMessageCallback([](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "Connected auth!" << std::endl;
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cout << "Error: " << msg->errorInfo.reason << std::endl;
        } else if (msg->type == ix::WebSocketMessageType::Close) {
            std::cout << "Closed auth!" << std::endl;
        }
    });
    webSocket.start();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    webSocket.stop();
    ix::uninitNetSystem();
    return 0;
}
