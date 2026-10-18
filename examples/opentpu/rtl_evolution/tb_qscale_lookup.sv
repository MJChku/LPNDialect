module tb_qscale_lookup;
  import otpu_fp::*;
  localparam int TW=18,NUM=131072,LM=FP_PIPE_MUL,LA=FP_PIPE_ADD;
  logic clk=0,rst=1,en=0,iv=0;
  always #5 clk=~clk;
  logic [TW-1:0] itag=0,oldtag,newtag;
  logic oldv,newv;
  f32_t amax=0,oldsc,oldinv,newsc,newinv;
  f32_t seen_sc[NUM],seen_inv[NUM];
  bit seen[NUM];
  int issued_at[NUM];
  int count=0,checked=0,cycles=0,enabled=0,held=0;
  otpu_qscale #(.TW(TW)) u_old (.clk,.rst,.en,.iv,.itag,.amax,.ov(oldv),.otag(oldtag),.sc(oldsc),.inv(oldinv));
  otpu_qscale_lookup #(.TW(TW)) u_new (.clk,.rst,.en,.iv,.itag,.amax,.ov(newv),.otag(newtag),.sc(newsc),.inv(newinv));
  always @(negedge clk) if (!rst) begin
    cycles++;
    if (en) begin
      enabled++;
      if (newv) begin
        if (newtag>=NUM || seen[newtag]) $fatal(1,"duplicate/out-of-range new scale tag");
        seen[newtag]=1;seen_sc[newtag]=newsc;seen_inv[newtag]=newinv;
        if (enabled-issued_at[newtag]!=4+LM) $fatal(1,"new scale latency %0d",enabled-issued_at[newtag]);
      end
      if (oldv) begin
        if (oldtag>=NUM || !seen[oldtag] || oldsc!==seen_sc[oldtag] || oldinv!==seen_inv[oldtag])
          $fatal(1,"scale mismatch tag=%0d sc old=%h new=%h inv old=%h new=%h",oldtag,oldsc,seen_sc[oldtag],oldinv,seen_inv[oldtag]);
        if (enabled-issued_at[oldtag]!=7*LM+4*LA+2) $fatal(1,"old scale latency %0d",enabled-issued_at[oldtag]);
        checked++;
      end
    end
    en=!(cycles%23>=7 && cycles%23<=11);
    if (!en) held++;
    iv=0;
    if (en && count<NUM) begin
      if (count<65536) begin
        logic [22:0] mantissa;
        case(count>>9)
          0:mantissa=0;
          1:mantissa=1;
          2:mantissa=23'h7fffff;
          3:mantissa=23'h400000;
          4:mantissa=23'h400001;
          default:mantissa=23'((count>>9)*32'd4194301);
        endcase
        amax={count[8],count[7:0],mantissa};
      end else amax=count*32'h9e3779b9;
      itag=TW'(count);iv=1;issued_at[count]=enabled;count++;
    end
    if (checked==NUM) begin
      $display("QSCALE_LOOKUP_GATE checked=%0d old_latency=%0d new_latency=%0d held=%0d",checked,7*LM+4*LA+2,4+LM,held);
      $finish;
    end
    if (cycles>NUM*2+100) $fatal(1,"qscale comparison timeout");
  end
  initial begin
    repeat(4) @(negedge clk);rst=0;
  end
endmodule
