module tb_recip_rom;
  import otpu_fp::*;
  parameter int PORTS=2;
  logic clk=0,rst=1,en=0;
  always #5 clk=~clk;
  logic [PORTS-1:0] iv,ov;
  logic [PORTS-1:0][22:0] fraction;
  logic [PORTS-1:0][7:0] itag,otag;
  logic [PORTS-1:0][31:0] value;
  logic [PORTS-1:0][31:0] expected [4];
  logic [PORTS-1:0][7:0] expected_tag [4];
  logic [PORTS-1:0] expected_valid [4];
  int unsigned count=0,cycles=0,checked=0,held=0;
  int unsigned limit=1<<23;
  otpu_recip_rom #(.PORTS(PORTS)) dut (.*);
  always @(posedge clk) begin
    if (rst) begin
      for (int t=0;t<4;t++) expected_valid[t]='0;
    end else if (en) begin
      for (int t=3;t>0;t--) begin
        expected[t]=expected[t-1];expected_tag[t]=expected_tag[t-1];expected_valid[t]=expected_valid[t-1];
      end
      for (int p=0;p<PORTS;p++) begin
        expected[0][p]=fp_recip({1'b0,8'd127,fraction[p]});
        expected_tag[0][p]=itag[p];
      end
      expected_valid[0]=iv;
    end
  end
  always @(negedge clk) begin
    if (!rst) begin
      for (int p=0;p<PORTS;p++) begin
        if (ov[p]!==expected_valid[3][p]) $fatal(1,"reciprocal valid mismatch");
        if (ov[p] && (value[p]!==expected[3][p] || otag[p]!==expected_tag[3][p]))
          $fatal(1,"reciprocal mismatch p=%0d count=%0d actual=%h expected=%h tag=%h expected_tag=%h",p,count,value[p],expected[3][p],otag[p],expected_tag[3][p]);
        if (ov[p] && en) checked++;
      end
      cycles++;
      en=(cycles%19!=3 && cycles%19!=4 && cycles%19!=5);
      if (!en) held++;
      iv='0;
      if (en) begin
        for (int p=0;p<PORTS;p++) if (count<limit) begin
          // Full sweep in a permutation: exercise both table ports and low /
          // high coefficient boundaries without sequential-address locality.
          fraction[p]=23'(count*23'd4194301);
          itag[p]=8'(count);iv[p]=1'b1;count++;
        end
      end
      if (count==limit && expected_valid[0]=='0 && expected_valid[1]=='0 && expected_valid[2]=='0 && expected_valid[3]=='0 && ov=='0) begin
        if (checked!=limit) $fatal(1,"reciprocal result count %0d != %0d",checked,limit);
        $display("RECIP_ROM_GATE checked=%0d clocks=%0d held=%0d ports=%0d",checked,cycles,held,PORTS);
        $finish;
      end
      if (cycles>limit*2+100) $fatal(1,"reciprocal gate timeout");
    end
  end
  initial begin
    iv='0;fraction='0;itag='0;
    void'($value$plusargs("limit=%d",limit));
    repeat(4) @(negedge clk);
    rst=0;en=1;
  end
endmodule
