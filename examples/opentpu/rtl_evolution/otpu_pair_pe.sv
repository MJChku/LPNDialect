// Two exact signed-int8 product streams sharing one operand and one multiplier.
// At most eight cascaded terms. Each field is separated by 18 bits; the sole
// signed-low-field overflow (eight copies of -128 * -128) has an explicit flag.
// TECH=0 is behavioral; TECH=1 instantiates exactly one UltraScale DSP48E2.
// This component is not connected to the MXU yet and does not prove FPGA fit.
module otpu_pair_pe #(
  parameter bit FIRST=0, LAST=0,
  parameter int TECH=0
) (
  input logic clk,en,
  input logic [7:0] common,lo0,lo1,
  input logic [47:0] pcin,
  input logic max_in,
  output logic [47:0] pcout,
  output logic max_out,
  output logic signed [23:0] p0,p1
);
  logic [7:0] common_r;
  logic max_product,max_prefix;
  logic [47:0] pf,pc;
  wire prefix=(FIRST || max_in) && max_product;
  always_ff @(posedge clk) if (en) begin
    common_r<=common;
    max_product<=common_r==8'h80 && lo0==8'h80;
  end
  if (LAST) assign max_prefix=prefix;
  else always_ff @(posedge clk) if (en) max_prefix<=prefix;
  if (TECH==0) begin : g_generic
    wire signed [26:0] packed_input=(27'($signed(lo1))<<<18)+27'($signed(lo0));
    logic signed [44:0] product;
    wire signed [47:0] total=(FIRST ? 48'sd0 : $signed(pcin))+48'(product);
    always_ff @(posedge clk) if (en) product<=45'(packed_input*$signed(common_r));
    if (LAST) assign pf=total;
    else always_ff @(posedge clk) if (en) pf<=total;
    assign pc=pf;
  end else begin : g_dsp48e2
    initial if (TECH!=1) $fatal(1,"pair PE supports behavioral or DSP48E2 only");
    DSP48E2 #(
      .A_INPUT("DIRECT"),.B_INPUT("DIRECT"),.AMULTSEL("AD"),.BMULTSEL("B"),.PREADDINSEL("A"),
      .USE_MULT("MULTIPLY"),.USE_WIDEXOR("FALSE"),.USE_SIMD("ONE48"),
      .AREG(0),.ACASCREG(0),.DREG(0),.ADREG(0),.BREG(1),.BCASCREG(1),.MREG(1),
      .PREG(LAST ? 0 : 1),.CREG(0),.INMODEREG(0),.OPMODEREG(0),.ALUMODEREG(0),
      .CARRYINREG(0),.CARRYINSELREG(0),.USE_PATTERN_DETECT("NO_PATDET"),
      .AUTORESET_PATDET("NO_RESET"),.MASK(48'h3fffffffffff),.PATTERN(48'h0),
      .SEL_MASK("MASK"),.SEL_PATTERN("PATTERN")
    ) u_dsp (
      .CLK(clk),.A({{22{lo0[7]}},lo0}),.D(27'($signed(lo1))<<<18),
      .B({{10{common[7]}},common}),.C(48'd0),.PCIN(pcin),
      .INMODE(5'b00100),.OPMODE(FIRST ? 9'b00_000_01_01 : 9'b00_001_01_01),
      .ALUMODE(4'b0000),.CARRYIN(1'b0),.CARRYINSEL(3'b000),
      .CEA1(1'b0),.CEA2(1'b0),.CED(1'b0),.CEAD(1'b0),.CEB1(en),.CEB2(en),
      .CEM(en),.CEP(LAST ? 1'b0 : en),.CEC(1'b0),.CEALUMODE(1'b0),.CECTRL(1'b0),
      .CECARRYIN(1'b0),.CEINMODE(1'b0),
      .RSTA(1'b0),.RSTB(1'b0),.RSTC(1'b0),.RSTD(1'b0),.RSTM(1'b0),.RSTP(1'b0),
      .RSTALLCARRYIN(1'b0),.RSTALUMODE(1'b0),.RSTCTRL(1'b0),.RSTINMODE(1'b0),
      .ACIN(30'd0),.BCIN(18'd0),.CARRYCASCIN(1'b0),.MULTSIGNIN(1'b0),
      .P(pf),.PCOUT(pc),.ACOUT(),.BCOUT(),.CARRYCASCOUT(),.MULTSIGNOUT(),.CARRYOUT(),
      .OVERFLOW(),.UNDERFLOW(),.PATTERNDETECT(),.PATTERNBDETECT(),.XOROUT());
  end
  wire low_overflow=max_prefix && pf[17];
  assign p0=$signed({{6{pf[17]}},pf[17:0]})+(low_overflow ? 24'sd262144 : 24'sd0);
  assign p1=24'($signed(pf)>>>18)+((pf[17] && !low_overflow) ? 24'sd1 : 24'sd0);
  assign pcout=LAST ? 48'd0 : pc;
  assign max_out=LAST ? 1'b0 : max_prefix;
endmodule
