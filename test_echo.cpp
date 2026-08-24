#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    ix::initNetSystem();
    ix::WebSocket webSocket;
    webSocket.setUrl("wss://ws.postman-echo.com/raw");
    webSocket.setOnMessageCallback([](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "Echo Connected!" << std::endl;
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cout << "Echo Error: " << msg->errorInfo.reason << std::endl;
        }
    });
    webSocket.start();
    std::this_thread::sleep_for(std::chrono::seconds(3));
    webSocket.stop();
    ix::uninitNetSystem();
    return 0;
}
