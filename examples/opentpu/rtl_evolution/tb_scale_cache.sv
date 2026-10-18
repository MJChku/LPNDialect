module tb_scale_cache #(parameter int LAT=8, ENTRIES=16);
  logic clk=0;
  always #5 clk=~clk;
  logic rst=1, flush=0, accept=0, ready, miss, idle, mem_req, mem_valid, rvalid;
  logic [31:0] addr=0,mem_addr;
  logic [255:0] mem_data,rdata,backing[64],responses[LAT],expected[LAT];
  logic [LAT-1:0] mv='0,ev='0;
  int received=0,requests=0,misses=0,hits=0,blocked=0,writes=0,cancelled=0;
  int write_line=-1;
  otpu_scale_cache #(.LAT(LAT),.ENTRIES(ENTRIES)) dut(.*);
  assign mem_valid=mv[LAT-1];
  assign mem_data=responses[LAT-1];
  always @(posedge clk) begin
    if (rst) begin cancelled+=$countones(ev[LAT-2:0]); mv<='0;ev<='0; end
    else begin
      mv[0]<=mem_req; ev[0]<=accept;
      if (mem_req) responses[0]<=backing[mem_addr>>3];
      if (accept) begin
        expected[0]<=backing[addr>>3]; requests++;
        if (miss) misses++; else hits++;
      end
      for (int k=1;k<LAT;k++) begin
        mv[k]<=mv[k-1]; responses[k]<=responses[k-1];
        ev[k]<=ev[k-1]; expected[k]<=expected[k-1];
      end
      if (write_line>=0) begin
        for (int w=0;w<8;w++) backing[write_line][w*32+:32]<=32'hd0000000+32'(writes*8+w);
        writes++;
      end
    end
    #1;
    if (rvalid!==ev[LAT-1]) $fatal(1,"response validity/order differs");
    if (rvalid) begin
      if (rdata!==expected[LAT-1]) $fatal(1,"response data differs at response %0d",received);
      received++;
    end
  end
  initial begin
    for (int i=0;i<64;i++) for (int w=0;w<8;w++) backing[i][32*w+:32]=32'(i*16+w);
    repeat(3) @(negedge clk);
    rst=0;
    for (int c=0;c<1000;c++) begin
      // Repeated pending hits, conflict misses and accepted-request bubbles.
      addr=32'(((c<80) ? (c/20) : (c>=90 && c<=110) ? 0 :
                (c%5==0) ? 16 : (c/9)%64)*8);
      flush=c==100 || c==180 || c==400 || c==700;
      rst=c==500;
      write_line=flush ? 0 : -1;
      #1;
      if (!ready && !flush) blocked++;
      accept=ready && (c%7!=3);
      @(negedge clk);
    end
    accept=0;flush=0;write_line=-1;
    repeat(LAT+3) @(negedge clk);
    if (!idle || requests!=received+cancelled || cancelled==0 || hits<100 || misses<20 || blocked==0 || writes!=4)
      $fatal(1,"incomplete coverage req=%0d responses=%0d hits=%0d misses=%0d blocked=%0d",requests,received,hits,misses,blocked);
    $display("SCALE_CACHE requests=%0d responses=%0d misses=%0d hits=%0d blocked=%0d writes=%0d cancelled=%0d",requests,received,misses,hits,blocked,writes,cancelled);
    $finish;
  end
endmodule
