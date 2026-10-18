// Standalone QACT/QST scale-unit candidate. The reciprocal's normalized
// Newton sequence is replaced by its exact table, retaining multiply/scale
// order and the original large/tiny/NaN cases. Selected by LOOKUP_SCALE.
module otpu_qscale_lookup
  import otpu_fp::*;
#(parameter int LM=FP_PIPE_MUL,TW=2,
  parameter string COEFF_FILE="",CORRECTION_FILE="") (
  input logic clk,rst,en,iv,
  input logic [TW-1:0] itag,
  input f32_t amax,
  output logic ov,
  output logic [TW-1:0] otag,
  output f32_t sc,inv
);
  logic normalized_valid;
  logic [TW+31:0] normalized_tag;
  f32_t normalized;
  otpu_recip_rom #(.PORTS(1),.TW(TW+32),.COEFF_FILE(COEFF_FILE),
      .CORRECTION_FILE(CORRECTION_FILE)) u_normalized (
    .clk,.rst,.en,.iv,.fraction(amax[22:0]),.itag({itag,1'b0,amax[30:0]}),
    .ov(normalized_valid),.value(normalized),.otag(normalized_tag));
  f32_t inv_product,sc_product;
  otpu_fmul #(.LAT(LM)) u_inv (.clk,.en,.a(F_127),.b(normalized),.y(inv_product));
  otpu_fmul #(.LAT(LM)) u_sc (.clk,.en,.a(normalized_tag[31:0]),.b(F_INV127),.y(sc_product));
  logic [LM-1:0] valid;
  logic [TW+31:0] tags [LM];
  always_ff @(posedge clk) begin
    if (rst) valid<='0;
    else if (en) begin
      valid[0]<=normalized_valid;
      for (int k=1;k<LM;k++) valid[k]<=valid[k-1];
    end
    if (en) begin
      tags[0]<=normalized_tag;
      for (int k=1;k<LM;k++) tags[k]<=tags[k-1];
    end
  end
  f32_t ax;
  logic [8:0] exponent_adjusted;
  assign ax=tags[LM-1][31:0];
  assign exponent_adjusted={1'b0,inv_product[30:23]}+rcp_ek(ax);
  assign ov=valid[LM-1];
  assign otag=tags[LM-1][TW+31:32];
  assign sc=ax==0 ? F_ZERO : sc_product;
  assign inv=(ax==0 || ax>=32'h7e800000) ? F_ZERO : exponent_adjusted>=9'd255 ? F_INF :
      {1'b0,exponent_adjusted[7:0],inv_product[22:0]};
endmodule
