// Numerical/flow-control gate for the actual quantizer, with synchronous TMEM.
module tb_quant_buffered;
  import otpu_pkg::*;
  parameter int XD=32;
  parameter bit STREAM_ACT=0;
  parameter bit LOOKUP_SCALE=0;
  localparam int LANES=64, WORDS=32768, DB=1<<20, AB=64, AR=16;
  logic clk=0, rst=1, start=0, ready, done, gnt, wr_idle;
  always #5 clk=~clk;
  cmd_t cmd;
  logic [LANES-1:0] re,re2,awe,vr;
  logic [LANES-1:0][31:0] ra,ra2,rd,rd2,va,vd;
  logic [LANES-1:0][3:0] vb;
  logic re3,aswe,dup,awant,areq,awe_d;
  logic [31:0] ra3,rd3,ai,aa,ad;
  logic [7:0] ar,asr,off;
  logic [15:0] asb;
  logic [63:0] asd;
  logic [LANES-1:0][7:0] av;
  logic [3:0] abe;
  logic [31:0] tmem[WORDS],dram[DB/4];
  logic [7:0] acts[AR][AB*64];
  logic [63:0] scales[AR][AB];
  logic [239:0] commands[128];
  longint cycles=0;
  int stall=0,ack=0;
  assign gnt=(stall==0) || ((stall==1) && cycles%11!=3 && cycles%11!=4)
      || ((stall==2) && !(cycles%71>=17 && cycles%71<34));
  assign wr_idle=ack==0;
  otpu_quant #(.D(64),.LANES(LANES),.STORE_LANES(LANES),.XD(XD)
`ifdef OTPU_HAS_STREAM_ACT
      ,.STREAM_ACT(STREAM_ACT)
`endif
`ifdef OTPU_HAS_LOOKUP_SCALE
      ,.LOOKUP_SCALE(LOOKUP_SCALE)
`endif
  ) dut (
    .clk,.rst,.start,.cmd,.rdy(ready),.done,.gnt,
    .t_ren(re),.t_raddr(ra),.t_rdata(rd),.t_ren2(re2),.t_raddr2(ra2),.t_rdata2(rd2),
    .t_ren3(re3),.t_raddr3(ra3),.t_rdata3(rd3),
    .act_we(awe),.act_row(ar),.act_idx(ai),.act_data(av),.act_dup(dup),.act_off(off),
    .asc_we(aswe),.asc_row(asr),.asc_blk(asb),.asc_data(asd),
    .a_want(awant),.wr_idle,.a_req(areq),.a_we(awe_d),.a_addr(aa),.a_wdata(ad),.a_be(abe),
    .v_req(vr),.v_addr(va),.v_data(vd),.v_be(vb),.pf_u(),.pf_frz());
  always_ff @(posedge clk) begin
    if (gnt) begin
      for (int l=0;l<LANES;l++) begin
        if (re[l]) rd[l]<=tmem[ra[l]];
        if (re2[l]) rd2[l]<=tmem[ra2[l]];
        if (vr[l] && !rst) begin
          if (va[l]>=DB/4) $fatal(1,"DRAM store out of range");
          for (int b=0;b<4;b++) if (vb[l][b]) dram[va[l]][8*b+:8]<=vd[l][8*b+:8];
        end
      end
      if (re3) rd3<=tmem[ra3];
      if (areq && awe_d && !rst)
        for (int b=0;b<4;b++) if (abe[b]) dram[aa][8*b+:8]<=ad[8*b+:8];
    end
    // ACT RAM is a separate physical port; its writes are accepted regardless
    // of TMEM grant. The quantizer gates its own output when necessary.
    if (!rst) begin
      for (int l=0;l<LANES;l++) if (awe[l]) begin
        acts[ar][ai+l]<=av[l];
        if (dup) acts[ar+off][ai+l]<=av[l];
      end
      if (aswe) begin
        scales[asr][asb]<=asd;
        if (dup) scales[asr+off][asb]<=asd;
      end
    end
    if (ack>0) ack<=ack-1;
    if (gnt && ((|vr) || (areq && awe_d))) ack<=3;
  end
  int row_left=0;
  logic [7:0] expect_row;
  logic [31:0] expect_idx;
  always @(posedge clk) begin
    if (rst || start) row_left=0;
    else if (STREAM_ACT && dut.bm && !dut.is_st) begin
      if (row_left!=0 && ((&awe)!=1'b1 || ar!=expect_row || ai!=expect_idx || !aswe))
        $fatal(1,"ACT row stream paused or reordered: remaining=%0d grant=%b",row_left,gnt);
      if (|awe) begin
        if (row_left==0) row_left=int'(dut.G/64);
        row_left--;
        expect_row=ar;
        expect_idx=ai+64;
      end
    end
  end
  string dir;
  int ncmd,fd;
  always @(negedge clk) begin
    cycles++;
    if (cycles>1000000) $fatal(1,"quantizer gate timeout");
  end
  initial begin
    longint begun;
    if (!$value$plusargs("dir=%s",dir)) $fatal(1,"missing dir");
    if (!$value$plusargs("ncmd=%d",ncmd)) $fatal(1,"missing ncmd");
    void'($value$plusargs("stall=%d",stall));
    for (int i=0;i<DB/4;i++) dram[i]=0;
    for (int r=0;r<AR;r++) begin
      for (int i=0;i<AB*64;i++) acts[r][i]=0;
      for (int i=0;i<AB;i++) scales[r][i]=0;
    end
    $readmemh({dir,"/tmem.hex"},tmem);
    $readmemh({dir,"/commands.hex"},commands);
    repeat(4) @(negedge clk); rst=0;
    for (int c=0;c<ncmd;c++) begin
      while (!ready) @(negedge clk);
      begun=cycles;
      cmd=cmd_t'(commands[c]);start=1;
      @(negedge clk);start=0;
      while (!done) @(negedge clk);
      $display("QCMD index=%0d cycles=%0d",c,cycles-begun);
      @(negedge clk);
      fd=$fopen($sformatf("%s/dram_%0d.bin",dir,c),"wb");
      for (int i=0;i<DB/4;i++) $fwrite(fd,"%u",dram[i]);
      $fclose(fd);
      fd=$fopen($sformatf("%s/act_%0d.bin",dir,c),"wb");
      for (int r=0;r<AR;r++) for (int i=0;i<AB*64;i++) $fwrite(fd,"%c",acts[r][i]);
      $fclose(fd);
      fd=$fopen($sformatf("%s/scales_%0d.bin",dir,c),"wb");
      for (int r=0;r<AR;r++) for (int i=0;i<AB;i++) begin
        $fwrite(fd,"%u",scales[r][i][31:0]);
        $fwrite(fd,"%u",scales[r][i][63:32]);
      end
      $fclose(fd);
    end
    $display("QUANT_GATE cycles=%0d",cycles);
    $finish;
  end
endmodule
