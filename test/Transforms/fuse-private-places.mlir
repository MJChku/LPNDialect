# RUN: lpn-opt --lpn-fuse-private-places %s | FileCheck %s

module {
  lpn.net {
    // Candidate place: single producer/consumer, no external visibility.
    lpn.place @buffer
    // Observable place should never be fused.
    lpn.place @external {observable}
    // Referenced via place_list, should be considered escaping.
    lpn.place @list_buf

    lpn.transition @producer {
      ^bb0:
        %buf = lpn.place_ref @buffer : !lpn.place
        %tok = lpn.token.create() : () -> !lpn.token
        %c0 = arith.constant 0.0 : f64
        lpn.emit %buf, %tok, %c0 : !lpn.place, !lpn.token, f64
        lpn.schedule.return
    }

    lpn.transition @consumer {
      ^bb0:
        %buf = lpn.place_ref @buffer : !lpn.place
        %tok = lpn.take %buf : !lpn.place -> !lpn.token
        %ext = lpn.place_ref @external : !lpn.place
        %c0 = arith.constant 0.0 : f64
        lpn.emit %ext, %tok, %c0 : !lpn.place, !lpn.token, f64
        lpn.schedule.return
    }

    lpn.transition @list_user {
      ^bb0:
        %p = lpn.place_ref @list_buf : !lpn.place
        %list = lpn.array %p : !lpn.array<!lpn.place>
        %zero = arith.constant 0 : index
        %handle = lpn.array.get %list, %zero : (!lpn.array<!lpn.place>, index) -> !lpn.place
        %tok = lpn.token.create() : () -> !lpn.token
        %delay = arith.constant 0.0 : f64
        lpn.emit %handle, %tok, %delay : !lpn.place, !lpn.token, f64
        lpn.schedule.return
    }

    lpn.halt
  }
}

# CHECK-LABEL: lpn.net
# CHECK: lpn.place @buffer {{.*}}lpn.fuse_candidate{{.*}}lpn.fuse_plan = {consumer = @consumer, producer = @producer}
# CHECK: lpn.place @external {observable}
# CHECK-NOT: lpn.fuse_candidate
