module tb_dram;
  logic clk = 0;
  always #5 clk = ~clk;
  logic a_req=0, a_we=0, sw_req=0, b_req=0, b_tag=0, b_we=0, dump=0;
  logic [31:0] a_addr=0,a_wdata=0,sw_addr=0,sw_wdata=0,b_addr=0;
  logic [3:0] a_be=0,sw_be=0;
  logic [31:0] b_wmask=0;
  logic [1023:0] b_wdata=0,b_rdata;
  logic [255:0] a_rdata;
  logic a_rvalid,b_rvalid,b_rtag,rd_idle;
  otpu_dram #(.WORDS(131072),.D(128),.LAT(8)) dut(.*);
  int cycle=0;
  always @(posedge clk) begin
    cycle <= cycle+1;
    if(a_rvalid) $display("READ A %0d %064h",cycle,a_rdata);
    if(b_rvalid) $display("READ B %0d %0d %0256h",cycle,b_rtag,b_rdata);
  end
  initial begin
    repeat(2) @(negedge clk);
    a_req=1; a_addr=65535;
    b_req=1; b_addr=65528; b_tag=1;
    @(negedge clk);
    a_req=0; b_req=0;
    repeat(2) @(negedge clk);
    sw_req=1; sw_addr=65536; sw_wdata=32'haabbccdd; sw_be=4'b0101;
    a_req=1; a_we=1; a_addr=65535; a_wdata=32'h11223344; a_be=4'b1010;
    @(negedge clk);
    sw_req=0; a_req=0; a_we=0;
    @(negedge clk);
    a_req=1; a_addr=65536;
    b_req=1; b_addr=65528; b_tag=0;
    @(negedge clk);
    a_req=0; b_req=0;
    repeat(2) @(negedge clk);
    b_req=1; b_we=1; b_addr=65520;
    for (int w=0;w<32;w++) begin
      b_wdata[32*w +:32]=32'h10203040+w;
      b_wmask[w]=(w%3)==0;
    end
    @(negedge clk);
    b_req=0; b_we=0;
    @(negedge clk);
    b_req=1; b_addr=65520;
    a_req=1; a_addr=65544;
    @(negedge clk);
    b_req=0; a_req=0;
    repeat(12) @(negedge clk);
    dump=1;
    @(negedge clk);
    dump=0;
    $finish;
  end
endmodule
