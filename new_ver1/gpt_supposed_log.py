def training(model, device='cuda', dtype=torch.float32,
             epoch=100, batch_size=32, lr=0.01, consist_depth=5,
             use_lr_scheduler=False, boundary_func=fun1):
    loss_list = []
    loss_func = nn.MSELoss()
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr,
                                 **({"fused": True} if "cuda" in str(device) else {}))
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=60, threshold=1e-4, cooldown=3)

    # 定义日志文件路径
    log_file_path = f'./training_logs/batch{batch_size}_epoch{epoch}_consist_depth{consist_depth}.txt'
    os.makedirs('./training_logs', exist_ok=True)  # 创建日志目录

    # 写入训练的初始配置
    write_log(log_file_path, f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(log_file_path, f"Hyperparameters: epoch={epoch}, batch_size={batch_size}, lr={lr}, consist_depth={consist_depth}")
    
    init_time = time.time()
    for i in range(epoch):
        loss_term = iteration(model, optimizer, loss_func, device, dtype, batch_size,
                              consist_depth=consist_depth, boundary_func=boundary_func)
        loss_list.append(loss_term)
        
        if (i + 1) % 20 == 0:  # 打印并记录日志
            log_message = f"Epoch: {i + 1}, Loss: {loss_term:.6f}, Time: {time.time() - init_time:.2f}s, LR: {optimizer.param_groups[0]['lr']}"
            print(log_message)
            write_log(log_file_path, log_message)  # 保存日志
        
        if (i + 1) % 50 == 0:  # 保存模型参数
            model_save_path = f'./model_parameter/batch{batch_size}_epoch{epoch}_checkpoint_epoch{i+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            write_log(log_file_path, f"Model checkpoint saved at: {model_save_path}")

        if use_lr_scheduler:
            scheduler.step(loss_term)

        # 提前终止
        if optimizer.param_groups[0]["lr"] <= 1.1e-8:
            write_log(log_file_path, "Learning rate dropped below threshold. Stopping early.")
            break

    # 训练结束日志
    write_log(log_file_path, f"Training ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(log_file_path, f"Final epoch: {i + 1}, Total time: {time.time() - init_time:.2f}s")
    
    # 绘制损失曲线
    if not debug:
        plt.plot(loss_list, label='loss')
        plt.legend()
        plt.savefig(f'./training_logs/loss_curve_batch{batch_size}_epoch{epoch}.png')
        write_log(log_file_path, f"Loss curve saved at: ./training_logs/loss_curve_batch{batch_size}_epoch{epoch}.png")
