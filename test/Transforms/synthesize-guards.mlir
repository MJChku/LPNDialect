// RUN: lpn-opt --lpn-synthesize-guards %s | FileCheck %s

module {
  lpn.net {
    lpn.place @a
    lpn.place @b
    lpn.place @out
    lpn.transition @nested {
      ^bb0:
        %a = lpn.place_ref @a : !lpn.place
        %outer = lpn.take %a : !lpn.place -> !lpn.token
        %cond = arith.constant true
        scf.if %cond {
          %b = lpn.place_ref @b : !lpn.place
          %inner = lpn.take %b : !lpn.place -> !lpn.token
          %out = lpn.place_ref @out : !lpn.place
          %zero = arith.constant 0.0 : f64
          lpn.emit %out, %inner, %zero : !lpn.place, !lpn.token, f64
        }
        %after = lpn.place_ref @out : !lpn.place
        %zero_after = arith.constant 0.0 : f64
        lpn.emit %after, %outer, %zero_after : !lpn.place, !lpn.token, f64
        lpn.schedule.return
    }
    lpn.transition @loop {
      ^bb0:
        %a = lpn.place_ref @a : !lpn.place
        %outer = lpn.take %a : !lpn.place -> !lpn.token
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 1 : index
        %c1 = arith.constant 1 : index
        scf.for %iv = %c0 to %c2 step %c1 {
          %b = lpn.place_ref @b : !lpn.place
          %inner = lpn.take %b : !lpn.place -> !lpn.token
          %out = lpn.place_ref @out : !lpn.place
          %zero = arith.constant 0.0 : f64
          lpn.emit %out, %inner, %zero : !lpn.place, !lpn.token, f64
        }
        %out = lpn.place_ref @out : !lpn.place
        %zero = arith.constant 0.0 : f64
        lpn.emit %out, %outer, %zero : !lpn.place, !lpn.token, f64
        lpn.schedule.return
    }
    lpn.transition @late_take {
      ^bb0:
        %a = lpn.place_ref @a : !lpn.place
        %first = lpn.take %a : !lpn.place -> !lpn.token
        %out = lpn.place_ref @out : !lpn.place
        %zero = arith.constant 0.0 : f64
        lpn.emit %out, %first, %zero : !lpn.place, !lpn.token, f64
        %b = lpn.place_ref @b : !lpn.place
        %later = lpn.take %b : !lpn.place -> !lpn.token
        lpn.schedule.return
    }
    lpn.halt
  }
}

// CHECK-LABEL: lpn.transition @nested
// CHECK: lpn.take {{.*}} {lpn.guard_id = 0 : i64}
// CHECK: lpn.take {{.*}} {lpn.guard_id = 1 : i64}
// CHECK: lpn.emit {{.*}} {lpn.guard_paths = {{\[\[0, 1\]\]}}}
// CHECK: lpn.emit {{.*}} {lpn.guard_paths = {{\[\[0, 1\]\]}}}
// CHECK-LABEL: lpn.transition @loop
// CHECK: lpn.take {{.*}} {lpn.guard_id = 0 : i64}
// CHECK: lpn.take {{.*}} {lpn.guard_id = 1 : i64}
// CHECK: lpn.emit {{.*}} {lpn.guard_paths = {{\[\[0, 1\]\]}}}
// CHECK: lpn.emit {{.*}} {lpn.guard_paths = {{\[\[0, 1\]\]}}}
// CHECK-LABEL: lpn.transition @late_take
// CHECK: lpn.take {{.*}} {lpn.guard_id = 0 : i64}
// CHECK: lpn.emit {{.*}} {lpn.guard_paths = {{\[\[0, 1\]\]}}}
// CHECK: lpn.take {{.*}} {lpn.guard_id = 1 : i64}
