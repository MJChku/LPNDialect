// Direct numerical gate for the actual VPU with one-cycle scratchpad reads.
module tb_vpu_wide;
  import otpu_pkg::*;
  parameter int LANES=64;
  localparam int WORDS=8192;
  logic clk=0, rst=1, start=0, ready, done, ren;
  always #5 clk=~clk;
  cmd_t cmd;
  logic [LANES-1:0] ae,be,we;
  logic [LANES-1:0][31:0] aa,ba,wa,ad,bd,wd;
  logic [31:0] mem [WORDS];
  logic [239:0] commands [64];
  logic [31:0] zeroes [LANES];
  initial for (int k=0;k<LANES;k++) zeroes[k]=0;
  otpu_vpu #(.LANES(LANES), .CL(LANES/4), .HAS_SE(1'b0)) dut (
    .clk,.rst,.start,.cmd,.rdy(ready),.done,.gnt(1'b1),.ren,.rhold(1'b0),
    .ta_en(ae),.ta_addr(aa),.ta_data(ad),.tb_en(be),.tb_addr(ba),.tb_data(bd),
    .tw_en(we),.tw_addr(wa),.tw_data(wd),.pf_u(),.pf_frz(),
    .ss_req(1'b0),.ss_gnt(),.ss_cfg('0),.ss_pe(1'b0),.ss_in_v(1'b0),.ss_in_d(zeroes),
    .ss_fk(3'd0),.ss_fi(5'd0),.ss_fd(zeroes),.ss_y_v(),.ss_y_d(),.ss_o_v(),.ss_o_d());
  always_ff @(posedge clk) begin
    for (int l=0;l<LANES;l++) begin
      if (ae[l]) ad[l]<=mem[aa[l]];
      if (be[l]) bd[l]<=mem[ba[l]];
      if (we[l] && !rst) mem[wa[l]]<=wd[l];
    end
  end
  string dir;
  int ncmd,fd;
  longint cycles=0;
  always @(negedge clk) begin
    cycles++;
    if (cycles>500000) $fatal(1,"wide VPU gate timeout");
  end
  initial begin
    longint command_start;
    if (!$value$plusargs("dir=%s",dir)) $fatal(1,"missing dir");
    if (!$value$plusargs("ncmd=%d",ncmd)) $fatal(1,"missing command count");
    $readmemh({dir,"/tmem.hex"},mem);
    $readmemh({dir,"/commands.hex"},commands);
    repeat(4) @(negedge clk);
    rst=0;
    for (int c=0;c<ncmd;c++) begin
      while (!ready) @(negedge clk);
      command_start=cycles;
      cmd=cmd_t'(commands[c]);start=1;
      @(negedge clk);start=0;
      while (!done) @(negedge clk);
      $display("VCMD index=%0d cycles=%0d",c,cycles-command_start);
      @(negedge clk);
    end
    repeat(4) @(negedge clk);
    fd=$fopen({dir,"/tmem_out.bin"},"wb");
    for (int k=0;k<WORDS;k++) $fwrite(fd,"%u",mem[k]);
    $fclose(fd);
    $display("VPU_GATE cycles=%0d",cycles);
    $finish;
  end
endmodule
