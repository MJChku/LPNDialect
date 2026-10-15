"""A component in another module, exercising scoped DSL helper calls."""

def connect_helper(net,source,destination,helper):
  @net.transition
  def forward():
    guard(count(source)>0)
    token = take(source)
    emit(destination,token.set("value",helper(token.get("value"))))
