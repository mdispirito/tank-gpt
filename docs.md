## Fine-tuning
This approach uses LoRA for fine-tuning to make it feasible on a Mac.

### MLX
For fine-tuning, we're using Apple's MLX-LM library. It handles all the complexities of LoRA behind the scenes.

In all system architectures, the most time intensive operations are moving memory between registers. Apple designed their silicon so that the CPU and GPU both have access to the same memory via the MMU (memory management unit). MLX is especially efficient because it takes advantage of this - transfers between the CPU and GPU are extremely fast.

### Training
The default number of training iterations is 1000.

Tweak the `iters` value in the [lora config](src/lora_config.yaml) to something like 100 when just testing things out and making sure everything works.