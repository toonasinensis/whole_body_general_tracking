
## CommandTerm

CommandTerm 对外暴露的接口为

1. Robot State 和 Reference Motion 的状态信息

行为上 CommandTerm 会在每个 env step 的最后

1.1. 根据 rewards 和 termination 的信息， 更新 CommandTerm 的状态

1.2. 根据 CommandTerm 的状态确定 Reference Motion

1.3. 重置 Robot State 

1.4. 更新相关的状态供其他模块消费

同时记录

1.5. metrics

1.6. motiom sampling

