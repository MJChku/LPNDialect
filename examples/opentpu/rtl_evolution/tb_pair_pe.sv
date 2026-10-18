`timescale 1ns/1ps
module tb_pair_pe;
  logic clk=0;
  always #5 clk=~clk;
  glbl glbl();
  logic en=1;
  logic [7:0] common=0,lo0=0,lo1=0;
  logic signed [23:0] single0[3],single1[3];
  logic chain_en=0;
  logic [7:0][7:0] chain_common='0,chain0='0,chain1='0;
  logic [7:0][47:0] cascade[4];
  logic [7:0] flags[2];
  logic signed [23:0] part0[3][8],part1[3][8];
  int single_checked=0,chain_checked=0,stalls=0,corner_checked=0;
  logic [31:0] random_state=32'h9105cb3e;
  for(genvar tech=0;tech<2;tech++) begin : g_single
    otpu_pair_pe #(.FIRST(1),.LAST(1),.TECH(tech)) u_pair (
      .clk,.en,.common,.lo0,.lo1,.pcin(48'd0),.max_in(1'b0),.pcout(),.max_out(),
      .p0(single0[tech]),.p1(single1[tech]));
  end
  otpu_pe #(.FIRST(1),.LAST(1)) u_ref0 (
    .clk,.en,.act(common),.wlo(lo0),.whi(lo0),.sel(1'b0),.pcin(48'd0),.pcout(),.p(single0[2]));
  otpu_pe #(.FIRST(1),.LAST(1)) u_ref1 (
    .clk,.en,.act(common),.wlo(lo1),.whi(lo1),.sel(1'b0),.pcin(48'd0),.pcout(),.p(single1[2]));
  for(genvar i=0;i<8;i++) begin : g_chain
    for(genvar tech=0;tech<2;tech++) begin : g_tech
      otpu_pair_pe #(.FIRST(i==0),.LAST(i==7),.TECH(tech)) u_pair (
        .clk,.en(chain_en),.common(chain_common[i]),.lo0(chain0[i]),.lo1(chain1[i]),
        .pcin(i==0 ? 48'd0 : cascade[tech][i==0 ? 0 : i-1]),
        .max_in(i==0 ? 1'b0 : flags[tech][i==0 ? 0 : i-1]),
        .pcout(cascade[tech][i]),.max_out(flags[tech][i]),.p0(part0[tech][i]),.p1(part1[tech][i]));
    end
    otpu_pe #(.FIRST(i==0),.LAST(i==7)) u_ref0 (
      .clk,.en(chain_en),.act(chain_common[i]),.wlo(chain0[i]),.whi(chain0[i]),.sel(1'b0),
      .pcin(i==0 ? 48'd0 : cascade[2][i==0 ? 0 : i-1]),.pcout(cascade[2][i]),.p(part0[2][i]));
    otpu_pe #(.FIRST(i==0),.LAST(i==7)) u_ref1 (
      .clk,.en(chain_en),.act(chain_common[i]),.wlo(chain1[i]),.whi(chain1[i]),.sel(1'b0),
      .pcin(i==0 ? 48'd0 : cascade[3][i==0 ? 0 : i-1]),.pcout(cascade[3][i]),.p(part1[2][i]));
  end
  task automatic check_single;
    #1;
    for(int tech=0;tech<2;tech++)
      if(single0[tech]!==single0[2] || single1[tech]!==single1[2])
        $fatal(1,"single mismatch tech=%0d case=%0d actual=%0d,%0d expected=%0d,%0d",tech,single_checked,
            single0[tech],single1[tech],single0[2],single1[2]);
    single_checked++;
  endtask
  task automatic check_chain;
    #1;
    for(int tech=0;tech<2;tech++) for(int i=0;i<8;i++)
      if(part0[tech][i]!==part0[2][i] || part1[tech][i]!==part1[2][i])
        $fatal(1,"chain mismatch tech=%0d pos=%0d case=%0d actual=%0d,%0d expected=%0d,%0d",tech,i,chain_checked,
            part0[tech][i],part1[tech][i],part0[2][i],part1[2][i]);
    chain_checked++;
    if(part0[2][7]==24'sd131072) corner_checked++;
  endtask
  initial begin
    // Let the vendor startup reset and all pipeline stages settle.
    repeat(40) @(negedge clk);
    for(int i=0;i<65536;i++) begin
      common=8'(i+1);lo0=8'(i>>8);lo1=8'(i^32'h5a);
      @(posedge clk);check_single();@(negedge clk);
    end
    if($test$plusargs("exhaustive")) begin
      common=0;repeat(3) @(negedge clk);
      for(int i=0;i<16777216;i++) begin
        // B is registered: prime it one clock ahead of the packed A/D.
        common=8'(i+1);lo0=8'(i>>8);lo1=8'(i>>16);
        @(posedge clk);check_single();@(negedge clk);
      end
    end
    en=0;chain_en=1;
    for(int c=0;c<100100;c++) begin
      chain_en=c%7!=0;
      if(!chain_en) stalls++;
      for(int i=0;i<8;i++) begin
        random_state={random_state[30:0],random_state[31]^random_state[21]^random_state[1]^random_state[0]};
        chain_common[i]=c<40 ? 8'h80 : c<80 ? 8'h80 : 8'(random_state);
        chain0[i]=c<40 ? 8'h80 : c<80 ? 8'h7f : 8'(random_state>>8);
        chain1[i]=c<80 ? 8'h80 : 8'(random_state>>16);
      end
      @(posedge clk);if(c>20) check_chain();@(negedge clk);
    end
    if(corner_checked==0 || stalls==0) $fatal(1,"corner or enable-stall coverage missing");
    $display("PAIR_PE single=%0d chain=%0d stalls=%0d overflow_corner=%0d",single_checked,chain_checked,stalls,corner_checked);
    $finish;
  end
endmodule
