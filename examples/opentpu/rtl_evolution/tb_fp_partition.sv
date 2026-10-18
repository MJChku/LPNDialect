// Independent NumPy golden values; verify every latency and enable hold.
module tb_fp_partition;
  import otpu_fp::*;
  logic clk=0,en=0;
  always #5 clk=~clk;
  f32_t a,b,c,d;
  f32_t add_y[6],mul_y[3],ma_y[3],mma_y;
  for (genvar k=0;k<6;k++) begin : g_add
    otpu_fadd #(.LAT(k+1)) dut(.clk,.en,.a,.b,.y(add_y[k]));
  end
  for (genvar k=0;k<3;k++) begin : g_mul
    otpu_fmul #(.LAT(k+1)) dut(.clk,.en,.a,.b,.y(mul_y[k]));
  end
  otpu_fmadd #(.LM(1),.LA(1)) ma0(.clk,.en,.a,.b,.c,.y(ma_y[0]));
  otpu_fmadd #(.LM(2),.LA(4)) ma1(.clk,.en,.a,.b,.c,.y(ma_y[1]));
  otpu_fmadd #(.LM(1),.LA(3)) ma2(.clk,.en,.a,.b,.c,.y(ma_y[2]));
  otpu_fmma #(.LM(1),.LA(1)) mma(.clk,.en,.a,.b,.c,.e(d),.y(mma_y));
  logic [127:0] inputs[8192],expected[8192];
  string dir;
  int n,issued=0,clocks=0;
  initial begin
    int j;
    f32_t held_a[6],held_m[3],held_ma[3],held_mma;
    if (!$value$plusargs("dir=%s",dir)) $fatal(1,"missing dir");
    if (!$value$plusargs("n=%d",n) || n>8192) $fatal(1,"invalid vector count");
    $readmemh({dir,"/inputs.hex"},inputs);
    $readmemh({dir,"/expected.hex"},expected);
    repeat(4) @(negedge clk);
    while (issued<n+8) begin
      @(negedge clk);
      en=clocks%13!=3 && clocks%13!=4 && clocks%13!=10;
      {a,b,c,d}=inputs[issued<n ? issued : n-1];
      held_a=add_y;held_m=mul_y;held_ma=ma_y;held_mma=mma_y;
      @(posedge clk);#1;
      if (en) begin
        for (int k=0;k<6;k++) begin
          j=issued-k;
          if (j>=0 && j<n && add_y[k]!==expected[j][127:96])
            $fatal(1,"ADD latency %0d vector %0d actual %h expected %h",k+1,j,add_y[k],expected[j][127:96]);
        end
        for (int k=0;k<3;k++) begin
          j=issued-k;
          if (j>=0 && j<n && mul_y[k]!==expected[j][95:64])
            $fatal(1,"MUL latency %0d vector %0d",k+1,j);
          j=issued-(k==0 ? 1 : k==1 ? 5 : 3);
          if (j>=0 && j<n && ma_y[k]!==expected[j][63:32])
            $fatal(1,"MADD configuration %0d vector %0d",k,j);
        end
        j=issued-1;
        if (j>=0 && j<n && mma_y!==expected[j][31:0]) $fatal(1,"MMA vector %0d",j);
        issued++;
      end else begin
        if (held_a!==add_y || held_m!==mul_y || held_ma!==ma_y || held_mma!==mma_y)
          $fatal(1,"disabled pipeline advanced");
      end
      clocks++;
    end
    $display("FP_PARTITION vectors=%0d clocks=%0d",n,clocks);
    $finish;
  end
endmodule
