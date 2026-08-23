`timescale 1ns / 1ps

/**
 * @file order_book_parser.v
 * @brief Simple Binary Encoding (SBE) L2 Order Book Parser.
 *
 * This module intercepts the raw payload of a UDP packet, scans
 * for the exchange's SBE Market Data Incremental Refresh message,
 * and extracts the Price and Quantity fields at hardware line-rate.
 */

module order_book_parser (
    input  wire         clk,
    input  wire         rst_n,
    
    // AXI4-Stream Interface from UDP Core
    input  wire [63:0]  s_axis_tdata,
    input  wire [7:0]   s_axis_tkeep,
    input  wire         s_axis_tvalid,
    input  wire         s_axis_tlast,
    output wire         s_axis_tready,
    
    // Extracted Fields Output (to DPDK or internal Memory)
    output reg  [63:0]  out_price,
    output reg  [63:0]  out_qty,
    output reg          out_is_bid,
    output reg          out_valid
);

    // State Machine definition
    localparam STATE_IDLE    = 3'd0;
    localparam STATE_HEADER  = 3'd1;
    localparam STATE_PAYLOAD = 3'd2;
    localparam STATE_WAIT    = 3'd3;
    
    reg [2:0]  state;
    reg [15:0] byte_count;
    
    assign s_axis_tready = 1'b1; // Always ready to sink data in this stub
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= STATE_IDLE;
            byte_count <= 16'd0;
            out_valid  <= 1'b0;
            out_price  <= 64'd0;
            out_qty    <= 64'd0;
            out_is_bid <= 1'b0;
        end else begin
            out_valid <= 1'b0; // Default clear
            
            case (state)
                STATE_IDLE: begin
                    if (s_axis_tvalid) begin
                        state      <= STATE_HEADER;
                        byte_count <= 16'd8; // First 8 bytes read
                    end
                end
                
                STATE_HEADER: begin
                    if (s_axis_tvalid) begin
                        byte_count <= byte_count + 16'd8;
                        // Example logic: Byte offset 16 contains message type
                        if (byte_count == 16'd16) begin
                            // Check if SBE Template ID == 1 (Incremental Refresh)
                            if (s_axis_tdata[15:0] == 16'h0001) begin
                                state <= STATE_PAYLOAD;
                            end else begin
                                state <= STATE_WAIT; // Ignore packet
                            end
                        end
                        if (s_axis_tlast) state <= STATE_IDLE;
                    end
                end
                
                STATE_PAYLOAD: begin
                    if (s_axis_tvalid) begin
                        byte_count <= byte_count + 16'd8;
                        // Example logic: Extract price and qty at specific byte offsets
                        if (byte_count == 16'd32) begin
                            out_price <= s_axis_tdata; // 64-bit little-endian price
                        end else if (byte_count == 16'd40) begin
                            out_qty <= s_axis_tdata;   // 64-bit little-endian qty
                        end else if (byte_count == 16'd48) begin
                            out_is_bid <= (s_axis_tdata[7:0] == 8'h01); // 1 = BID, 2 = ASK
                            out_valid  <= 1'b1;         // Signal valid extraction
                            state      <= STATE_WAIT;
                        end
                        
                        if (s_axis_tlast) state <= STATE_IDLE;
                    end
                end
                
                STATE_WAIT: begin
                    // Consume rest of packet
                    if (s_axis_tvalid && s_axis_tlast) begin
                        state <= STATE_IDLE;
                    end
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
