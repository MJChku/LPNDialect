// Per-MXU 32-byte read-through cache. Repeated requests coalesce even before
// the first response. Entries cannot be replaced until all accepted readers
// retire, so a write invalidation cannot change an older request's snapshot.
// Hits retain the backing port's fixed latency and response order.
module otpu_scale_cache #(
  parameter int LAT=8, ENTRIES=16
) (
  input logic clk, rst, flush,
  input logic accept,
  input logic [31:0] addr, // word address, independent of grant
  output logic ready, miss, idle,
  output logic mem_req,
  output logic [31:0] mem_addr,
  input logic mem_valid,
  input logic [255:0] mem_data,
  output logic rvalid,
  output logic [255:0] rdata
);
  localparam int IW=$clog2(ENTRIES), RW=$clog2(LAT+1);
  initial if (LAT<2 || ENTRIES<2 || (ENTRIES&(ENTRIES-1))!=0)
    $fatal(1,"scale cache requires LAT>=2 and power-of-two ENTRIES>=2");
  logic [ENTRIES-1:0] valid;
  logic [28:0] tags [ENTRIES];
  logic [RW-1:0] refs [ENTRIES];
  // One fill write and one registered read, with explicit fill forwarding.
  logic [255:0] data [ENTRIES];
  logic [255:0] cached_read;
  logic [LAT-1:0] live, is_miss;
  logic [IW-1:0] slots [LAT];
  wire [IW-1:0] index=addr[3+:IW];
  wire hit=valid[index] && tags[index]==addr[31:3];
  wire fill=mem_valid && live[LAT-1] && is_miss[LAT-1];
  assign ready=!rst && !flush && (hit || refs[index]==0);
  assign miss=!hit;
  assign mem_req=accept && ready && miss;
  assign mem_addr=addr;
  assign rvalid=live[LAT-1];
  assign rdata=is_miss[LAT-1] ? mem_data : cached_read;
  always_comb begin
    idle=1'b1;
    for (int i=0;i<ENTRIES;i++) if (refs[i]!=0) idle=1'b0;
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      valid<='0; live<='0; is_miss<='0;
      for (int i=0;i<ENTRIES;i++) refs[i]<='0;
    end else begin
      live[0]<=accept && ready;
      is_miss[0]<=miss;
      slots[0]<=index;
      for (int i=1;i<LAT;i++) begin
        live[i]<=live[i-1]; is_miss[i]<=is_miss[i-1]; slots[i]<=slots[i-1];
      end
      if (flush) valid<='0;
      if (mem_req) begin valid[index]<=1'b1; tags[index]<=addr[31:3]; end
      if (fill) data[slots[LAT-1]]<=mem_data;
      if (fill && slots[LAT-1]==slots[LAT-2]) cached_read<=mem_data;
      else cached_read<=data[slots[LAT-2]];
      for (int i=0;i<ENTRIES;i++) begin
        case ({accept && ready && index==IW'(i),live[LAT-1] && slots[LAT-1]==IW'(i)})
          2'b10: refs[i]<=refs[i]+1'b1;
          2'b01: refs[i]<=refs[i]-1'b1;
          default: ;
        endcase
      end
    end
  end
`ifndef SYNTHESIS
  always @(posedge clk) if (!rst) begin
    if (accept && !ready) $fatal(1,"scale cache accepted while blocked");
    if (live[LAT-1] && is_miss[LAT-1] && !mem_valid)
      $fatal(1,"scale cache backing latency differs");
  end
`endif
endmodule
