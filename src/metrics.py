def get_residual_map(test_data):
    test_img = test_data[0].to(device)
    rec_test_img = model(test_img)
    
    l1_residual_map = np.mean(np.abs(np.squeeze(test_img.cpu().detach().numpy(), axis=0) - np.squeeze(rec_test_img.cpu().detach().numpy(), axis=0)), axis=0)
    l2_residual_map = np.linalg.norm(np.squeeze(test_img.cpu().detach().numpy(), axis=0) - np.squeeze(rec_test_img.cpu().detach().numpy(), axis=0), axis=0)
    ssim_residual_map = ssim(np.squeeze(test_img.cpu().detach().numpy(), axis=0).transpose(1, 2, 0), np.squeeze(rec_test_img.cpu().detach().numpy(), axis=0).transpose(1, 2, 0), win_size=11, full=True, multichannel=True)[1]
    ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)

#     fig = plt.figure(figsize=(12, 4))
    
#     ax1 = fig.add_subplot(1, 3, 1)
#     plt.imshow(l1_residual_map)

#     ax2 = fig.add_subplot(1, 3, 2)
#     plt.imshow(l2_residual_map)
    
#     ax3 = fig.add_subplot(1, 3, 3)
#     plt.imshow(ssim_residual_map)
    
    return l1_residual_map, l2_residual_map, ssim_residual_map
  
  
  def get_max_threshold():
    good = DataLoader(validate_hazelnut, batch_size=1, shuffle=True, num_workers=0)
    total_rec_l1, total_rec_l2, total_rec_ssim = [], [], []

    for i, val_data in enumerate(good):
        l1_residual_map, l2_residual_map, ssim_residual_map = get_residual_map(val_data)
        total_rec_l1.append(l1_residual_map)
        total_rec_l2.append(l2_residual_map)
        total_rec_ssim.append(ssim_residual_map)


    total_rec_l1 = np.array(total_rec_l1)
    total_rec_l2 = np.array(total_rec_l2)
    total_rec_ssim = np.array(total_rec_ssim)

    l1_threshold = float(np.max(total_rec_l1))
    l2_threshold = float(np.max(total_rec_l2))
    ssim_threshold = float(np.max(total_rec_ssim))

    return l1_threshold, l2_threshold, ssim_threshold

  
def get_p_threshold(percentage):
    good = DataLoader(validate_hazelnut, batch_size=1, shuffle=True, num_workers=0)
    total_rec_l1, total_rec_l2, total_rec_ssim = [], [], []

    for i, val_data in enumerate(good):
        l1_residual_map, l2_residual_map, ssim_residual_map = get_residual_map(val_data)
        total_rec_l1.append(l1_residual_map)
        total_rec_l2.append(l2_residual_map)
        total_rec_ssim.append(ssim_residual_map)


    total_rec_l1 = np.array(total_rec_l1)
    total_rec_l2 = np.array(total_rec_l2)
    total_rec_ssim = np.array(total_rec_ssim)

    l1_threshold = float(np.percentile(total_rec_l1, percentage))
    l2_threshold = float(np.percentile(total_rec_l2, percentage))
    ssim_threshold = float(np.percentile(total_rec_ssim, percentage))

    return l1_threshold, l2_threshold, ssim_threshold
  
  
 def pixel_level_metrics(ground_truth, mask):
    TP = FP = TN = FN = 0
    for i in range(len(ground_truth)):
        for j in range(len(ground_truth[i])):
            if ground_truth[i][j] == 1 and mask[i][j] == 1: TP += 1
            elif ground_truth[i][j] == 0 and mask[i][j] == 1: FP += 1
            elif ground_truth[i][j] == 1 and mask[i][j] == 0: TN += 1
            else: FN += 1
                
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    PRC = TP / (TP + FP)
    IoU = TP / (TP + FP + FN)
    
    return TP, FP, TN, FN, TPR, FPR, PRC, IoU
  
  
 for i, test_data in enumerate(test_hazelnut_dataloader):
    if i == 0:
        l1_residual_map, l2_residual_map, ssim_residual_map = get_residual_map(test_data)

        percentage = 99.95
#         l1_threshold, l2_threshold, ssim_threshold = get_max_threshold()
        l1_threshold, l2_threshold, ssim_threshold = get_p_threshold(percentage)

        mask = np.zeros((256, 256))
        mask[l1_residual_map > l1_threshold] = 1
        mask[l2_residual_map > l2_threshold] = 1
        mask[ssim_residual_map > ssim_threshold] = 1 
        
        plt.imshow(mask, cmap='gray')
        
        TP, FP, TN, FN, TPR, FPR, PRC, IoU = pixel_level_metrics(test_data[2][0][0], mask)
        
        break
