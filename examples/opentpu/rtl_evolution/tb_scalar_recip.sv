// Direct numerical gate for the actual VPU with one-cycle scratchpad reads.
module tb_scalar_recip;
  import otpu_pkg::*;
  parameter int LANES=64;
  localparam int WORDS=32768;
  parameter int ROWS=4;
  parameter bit LOOKUP=1;
  parameter int BUFFERED=0;
  logic clk=0, rst=1, start=0, ready, done, ren,gnt;
  int stall=0,pipeline=0,completed=0;
  assign gnt=stall==0 || (stall==1 && cycles%13!=3 && cycles%13!=4) || (stall==2 && !(cycles%71>=17 && cycles%71<34));
  always #5 clk=~clk;
  cmd_t cmd;
  logic [LANES-1:0] ae,be,we;
  logic [LANES-1:0][31:0] aa,ba,wa,ad,bd,wd;
  logic [31:0] mem [WORDS];
  logic [239:0] commands [128];
  logic [31:0] zeroes [LANES];
  initial for (int k=0;k<LANES;k++) zeroes[k]=0;
  otpu_vpu #(.LANES(LANES), .CL(LANES/4), .HAS_SE(1'b0), .RED_ROWS(ROWS), .WBUF(BUFFERED), .RECIP_SCALAR_LOOKUP(LOOKUP)) dut (
    .clk,.rst,.start,.cmd,.rdy(ready),.done,.gnt(gnt),.ren,.rhold(1'b0),
    .ta_en(ae),.ta_addr(aa),.ta_data(ad),.tb_en(be),.tb_addr(ba),.tb_data(bd),
    .tw_en(we),.tw_addr(wa),.tw_data(wd),.pf_u(),.pf_frz(),
    .ss_req(1'b0),.ss_gnt(),.ss_cfg('0),.ss_pe(1'b0),.ss_in_v(1'b0),.ss_in_d(zeroes),
    .ss_fk(3'd0),.ss_fi(5'd0),.ss_fd(zeroes),.ss_y_v(),.ss_y_d(),.ss_o_v(),.ss_o_d());
  always_ff @(posedge clk) begin
    for (int l=0;l<LANES;l++) begin
      if (ren && ae[l]) ad[l]<=mem[aa[l]];
      if (ren && be[l]) bd[l]<=mem[ba[l]];
      if (gnt && we[l] && !rst) mem[wa[l]]<=wd[l];
    end
  end
  string dir;
  int ncmd,fd;
  longint cycles=0;
  always @(negedge clk) begin
    cycles++;
    if (!rst && done) completed++;
    if (cycles>500000) $fatal(1,"wide VPU gate timeout");
  end
  initial begin
    longint command_start;
    if (!$value$plusargs("dir=%s",dir)) $fatal(1,"missing dir");
    if (!$value$plusargs("ncmd=%d",ncmd)) $fatal(1,"missing command count");
    void'($value$plusargs("stall=%d",stall));
    void'($value$plusargs("pipeline=%d",pipeline));
    $readmemh({dir,"/tmem.hex"},mem);
    $readmemh({dir,"/commands.hex"},commands);
    repeat(4) @(negedge clk);
    rst=0;
    for (int c=0;c<ncmd;c++) begin
      while (!ready) @(negedge clk);
      command_start=cycles;
      cmd=cmd_t'(commands[c]);start=1;
      @(negedge clk);start=0;
      if (!pipeline) begin
      while (!done) @(negedge clk);
      $display("VCMD index=%0d cycles=%0d",c,cycles-command_start);
      @(negedge clk);
      fd=$fopen($sformatf("%s/tmem_%0d.bin",dir,c),"wb");
      for (int k=0;k<WORDS;k++) $fwrite(fd,"%u",mem[k]);
      $fclose(fd);
      end
    end
    while (completed<ncmd) @(negedge clk);
    repeat(4) @(negedge clk);
    fd=$fopen({dir,"/tmem_out.bin"},"wb");
    for (int k=0;k<WORDS;k++) $fwrite(fd,"%u",mem[k]);
    $fclose(fd);
    $display("SCALAR_RECIP_GATE cycles=%0d completed=%0d",cycles,completed);
    $finish;
  end
endmodule
