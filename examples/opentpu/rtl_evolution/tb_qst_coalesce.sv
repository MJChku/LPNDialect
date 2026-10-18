module tb_qst_coalesce;
  localparam int L=64;
  logic [L-1:0] want,req,want_o,req_o;
  logic [L-1:0][31:0] addr,data,addr_o,data_o;
  logic [L-1:0][3:0] be,be_o;
  logic [31:0] original[512],merged[512];
  int checked=0,raw_words=0,merged_words=0;
  otpu_qst_coalesce #(.L(L)) dut(.*);
  initial begin
    for (int mode=0;mode<5;mode++)
      for (int off=0;off<4;off++)
        for (int tail=0;tail<6;tail++)
          for (int select_req=0;select_req<3;select_req++)
            for (int holes=0;holes<2;holes++) begin
              int n,stride;
              case(tail) 0:n=1;1:n=2;2:n=3;3:n=15;4:n=63;default:n=64;endcase
              case(mode) 0:stride=1;1:stride=4;2:stride=17;3:stride=0;default:stride=1;endcase
              want='0;req='0;addr='0;data='0;be='0;
              for (int l=0;l<L;l++) begin
                int byteaddr;
                byteaddr=off+l*stride;
                want[l]=l<n && (!holes || l%5!=1);
                req[l]=want[l] && (select_req==1 || (select_req==2 && l%2==1));
                addr[l]=32'(byteaddr/4);
                be[l]=(mode==4) ? 4'b0011 : (4'b0001<<(byteaddr%4));
                data[l]=32'h12345678 ^ 32'(l*113+off*9);
              end
              #1;
              for (int w=0;w<512;w++) begin original[w]=32'hdeadbeef;merged[w]=32'hdeadbeef;end
              for (int l=0;l<L;l++) begin
                if (req[l]) for (int b=0;b<4;b++) if (be[l][b]) original[addr[l]][8*b+:8]=data[l][8*b+:8];
                if (req_o[l]) for (int b=0;b<4;b++) if (be_o[l][b]) merged[addr_o[l]][8*b+:8]=data_o[l][8*b+:8];
              end
              for (int w=0;w<512;w++) if (original[w]!==merged[w])
                $fatal(1,"word data mismatch mode=%0d off=%0d tail=%0d req=%0d holes=%0d word=%0d",mode,off,tail,select_req,holes,w);
              if (mode==0 && !holes && select_req==1 && int'($countones(req_o))!=(off+n+3)/4)
                $fatal(1,"packed transaction count differs");
              if (((mode==1 || mode==2) && $countones(want)>1) || mode==4)
                if (req_o!==req || be_o!==be || data_o!==data || addr_o!==addr)
                  $fatal(1,"scatter or unsupported mask changed");
              checked++;raw_words+=$countones(req);merged_words+=$countones(req_o);
            end
    if (checked!=720 || merged_words>=raw_words) $fatal(1,"incomplete gate");
    $display("QST_COALESCE checked=%0d raw_words=%0d merged_words=%0d",checked,raw_words,merged_words);
    $finish;
  end
endmodule
