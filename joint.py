import numpy as np
import tensorflow as tf
from tqdm import tqdm 

temperature = 100 

def prepare_dataset(x, x_flat, y, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, x_flat, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def train(model_f, model_g, x_train, x_train_flat, y_train, dataset_val, epochs, batch_size):

    dataset_train = prepare_dataset(x_train, x_train_flat, y_train, batch_size)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # 设置学习率调度器
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=3e-3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    accuracy_metric_f = tf.keras.metrics.CategoricalAccuracy()
    accuracy_metric_g = tf.keras.metrics.CategoricalAccuracy()


    for epoch in range(epochs):
        # 根据轮次动态调整损失函数的系数
        imbd_f = 1.4
        lmbd_g = 0.6
        lmbd_of = 0.8
        gama_fef = 0.5

        
        print(f"Epoch {epoch + 1}/{epochs}")
        total_samples = 0
        progbar = tf.keras.utils.Progbar(target=len(x_train), unit_name='samples')

        for (x_batch, x_batch_flat, y_batch) in dataset_train:
            with tf.GradientTape() as tape:
                predictions_f = model_f(x_batch, training=True)
                predictions_g = model_g(x_batch_flat, training=True)

                loss_f = loss_fn(y_batch, predictions_f)
                loss_g = model_g.calculate_tree_loss(y_batch, predictions_g)
                loss_fef = Loss_fef(model_f, model_g, x_batch_flat, x_batch)
                loss_of = Loss_of(model_f, model_g, x_batch_flat, x_batch, temperature=temperature)
                loss_all = imbd_f * loss_f + lmbd_g * loss_g + gama_fef * loss_fef + lmbd_of * loss_of

            grads = tape.gradient(loss_all, model_f.trainable_variables + model_g.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_f.trainable_variables + model_g.trainable_variables))
            
            accuracy_metric_f.update_state(y_batch, predictions_f)
            accuracy_metric_g.update_state(y_batch, predictions_g)

            total_samples += len(x_batch)
            progbar.update(total_samples, values=[("loss_f", loss_f.numpy()), ("loss_g", loss_g.numpy()), 
                                         ("accuracy_f", accuracy_metric_f.result().numpy()),
                                         ("accuracy_g", accuracy_metric_g.result().numpy())])
    
    return model_f, model_g

def softmax_temperature(logits, temperature):
    """使用特定温度参数调整 logits 并应用 softmax 得到概率分布。"""
    return tf.nn.softmax(logits / temperature)

def Loss_of(f, g, x_flat, x, temperature):
    
    # 获取预测器和解释器的 logits
    v_i_logits = f.get_logits(x)
    z_i_logits = g.get_logits(x_flat)
    
    # 使用温度参数调整 logits 并计算概率
    q_i = softmax_temperature(z_i_logits, temperature)
    p_i = softmax_temperature(v_i_logits, temperature)
    
    # 计算损失
    loss = -tf.reduce_sum(q_i * tf.math.log(p_i))
    return loss

def Loss_fef(f, g, x_flat, x):
    # 获取预测器和解释器的 logits
    v_i_logits = f.get_logits(x)
    z_i_logits = g.get_logits(x_flat)
    # 计算平方差损失
    mse = tf.keras.losses.MeanSquaredError()
    loss_fef = mse(z_i_logits, v_i_logits)
    return loss_fef

def evaluate(model_f, model_g, x, x_flat, y):
    accuracy_metric_f = tf.keras.metrics.CategoricalAccuracy()
    accuracy_metric_g = tf.keras.metrics.CategoricalAccuracy()
         
    predictions_f = model_f(x, training=False)
    predictions_g = model_g(x_flat, training=False)
 
    accuracy_metric_f.update_state(y, predictions_f)
    accuracy_metric_g.update_state(y, predictions_g)
    
    accuracy_f = accuracy_metric_f.result().numpy()
    accuracy_g = accuracy_metric_g.result().numpy()
    
    accuracy_metric_f.reset_states()
    accuracy_metric_g.reset_states()
    
    print(f'Accuracy of model_f: {accuracy_f * 100:.2f}%')
    print(f'Accuracy of model_g: {accuracy_g * 100:.2f}%')
    
    return accuracy_f, accuracy_g


def analyze(f, g, x_test, x_test_flat, y_test):
    # 初始化混淆矩阵
    conf_matx_fy = np.zeros([2, 2])  # 对于f模型和真实标签y的混淆矩阵
    conf_matx_fg = np.zeros([2, 2])  # 对于f模型和g模型的混淆矩阵，用于计算保真度
    conf_matx_gy = np.zeros([2, 2])  # 对于g模型和真实标签y的混淆矩阵，用于计算g模型的准确度

    # 对整个数据集进行预测
    output_f = f.predict(x_test)
    output_g = g.predict(x_test_flat)
    y = np.argmax(y_test, axis=1)
    pred_f = np.argmax(output_f, axis=1)
    pred_g = np.argmax(output_g, axis=1)

    # 更新混淆矩阵
    for i in range(len(y)):
        conf_matx_fy[y[i], pred_f[i]] += 1
        conf_matx_fg[pred_f[i], pred_g[i]] += 1
        conf_matx_gy[y[i], pred_g[i]] += 1

    # 计算准确度和保真度
    accuracy_f = np.diag(conf_matx_fy).sum() / len(y)  
    fidelity = np.diag(conf_matx_fg).sum() / len(y)  
    accuracy_g = np.diag(conf_matx_gy).sum() / len(y)  

    return accuracy_f, fidelity, accuracy_g
