// Merge consecutive byte stores into real byte-enabled word transactions.
// Packed detection uses wants/addresses, independent of downstream grant.
// Scatter, repeated addresses and unexpected masks retain the original ports.
module otpu_qst_coalesce #(parameter int L=64) (
  input logic [L-1:0] want, req,
  input logic [L-1:0][31:0] addr, data,
  input logic [L-1:0][3:0] be,
  output logic [L-1:0] want_o, req_o,
  output logic [L-1:0][31:0] addr_o, data_o,
  output logic [L-1:0][3:0] be_o
);
  logic packed_bytes;
  int unsigned offset;
  always_comb begin
    packed_bytes=want[0]; offset=0;
    case (be[0])
      4'b0001: offset=0;
      4'b0010: offset=1;
      4'b0100: offset=2;
      4'b1000: offset=3;
      default: packed_bytes=1'b0;
    endcase
    for (int l=0;l<L;l++) if (want[l])
      if (addr[l]!=addr[0]+32'((offset+l)/4) || be[l]!=(4'b0001<<((offset+l)%4)))
        packed_bytes=1'b0;
    want_o=want; req_o=req; addr_o=addr; data_o=data; be_o=be;
    if (packed_bytes) begin
      want_o='0; req_o='0; data_o='0; be_o='0;
      for (int l=0;l<L;l++) begin
        logic owner;
        owner=want[l];
        for (int k=1;k<=3;k++)
          if (l>=k && want[l-k] && addr[l-k]==addr[l]) owner=1'b0;
        if (owner) begin
          want_o[l]=1'b1;
          for (int k=0;k<=3;k++) if (l+k<L && req[l+k] && addr[l+k]==addr[l]) begin
            req_o[l]=1'b1;
            for (int b=0;b<4;b++) if (be[l+k][b]) begin
              be_o[l][b]=1'b1;
              data_o[l][8*b+:8]=data[l+k][8*b+:8];
            end
          end
        end
      end
    end
  end
endmodule
