from types import SimpleNamespace

classes = ['pothole', 'expand', 'crack']
ckpt = '../ckpt/ph_ckpt.pth'
train = SimpleNamespace(
    train_img_path='../dataset/road/crack_ph/train/*.jpg',
    val_img_path='../dataset/road/crack_ph/val/*.jpg',
    save_path = '../ph_model_weight/UPP_ckpt_effb6',
    max_lr = 5e-4,
    epochs = 80,
    classes = classes
    )
eval = SimpleNamespace(
    image_path = '../dataset/road/crack_ph/val/*.jpg',
    save_path = '../output/performance/crack_train.json',
    ckpt = ckpt,
    classes = classes
    )
prediction = SimpleNamespace(
    image_path = '../dataset/road/20230530/100FTASK/*.JPG',
    save_path = '../output/prediction_c_p/',
    ckpt = ckpt,
    classes = classes
    )