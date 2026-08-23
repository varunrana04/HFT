`timescale 1ns / 1ps

/**
 * @file udp_rx_core.v
 * @brief Top-level wrapper for Gigabit Ethernet UDP Reception.
 *
 * This module takes the raw MAC/PHY AXI4-Stream, strips the Ethernet,
 * IPv4, and UDP headers, verifies the destination port matches our HFT
 * market data port, and forwards the payload to the order_book_parser.
 */

module udp_rx_core (
    input  wire         clk,
    input  wire         rst_n,
    
    // MAC Interface (Input)
    input  wire [63:0]  mac_axis_tdata,
    input  wire [7:0]   mac_axis_tkeep,
    input  wire         mac_axis_tvalid,
    input  wire         mac_axis_tlast,
    output wire         mac_axis_tready,
    
    // Extracted Fields (Output to PCIe / CPU)
    output wire [63:0]  parsed_price,
    output wire [63:0]  parsed_qty,
    output wire         parsed_is_bid,
    output wire         parsed_valid
);

    // Target UDP Port for Market Data (e.g. 50000)
    localparam TARGET_UDP_PORT = 16'd50000;
    
    // Internal signals for payload extraction
    wire [63:0] payload_tdata;
    wire [7:0]  payload_tkeep;
    wire        payload_tvalid;
    wire        payload_tlast;
    wire        payload_tready;
    
    // In a full implementation, a UDP/IP stripping module would sit here.
    // For this stub, we wire the MAC stream directly to the payload stream,
    // assuming the parser will just offset the byte counts appropriately.
    assign payload_tdata   = mac_axis_tdata;
    assign payload_tkeep   = mac_axis_tkeep;
    assign payload_tvalid  = mac_axis_tvalid;
    assign payload_tlast   = mac_axis_tlast;
    assign mac_axis_tready = payload_tready;

    // Instantiate the SBE Order Book Parser
    order_book_parser u_parser (
        .clk            (clk),
        .rst_n          (rst_n),
        
        .s_axis_tdata   (payload_tdata),
        .s_axis_tkeep   (payload_tkeep),
        .s_axis_tvalid  (payload_tvalid),
        .s_axis_tlast   (payload_tlast),
        .s_axis_tready  (payload_tready),
        
        .out_price      (parsed_price),
        .out_qty        (parsed_qty),
        .out_is_bid     (parsed_is_bid),
        .out_valid      (parsed_valid)
    );

endmodule
