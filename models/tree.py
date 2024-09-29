import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
import os

class Node(Layer):
    def __init__(self, depth, max_depth, n_features, n_classes, 
                 penalty_strength, penalty_decay, inv_temp=1.0, **kwargs):
        super(Node, self).__init__(**kwargs)
        self.depth = depth
        self.is_leaf = depth == max_depth
        self.inv_temp = inv_temp
        self.penalty_strength = penalty_strength * (penalty_decay ** depth)

        if not self.is_leaf:
            self.w = self.add_weight(name=f'weight_depth_{depth}', shape=(n_features, 1),
                                     initializer='random_normal', trainable=True)
            self.b = self.add_weight(name=f'bias_depth_{depth}', shape=(1,),
                                     initializer='zeros', trainable=True)
        else:
            self.phi = self.add_weight(name=f'phi_depth_{depth}', shape=(n_features, n_classes),
                                       initializer='random_normal', trainable=True)
            self.phi_b = self.add_weight(name=f'phi_bias_depth_{depth}', shape=(n_classes,),
                                         initializer='zeros', trainable=True)

    def call(self, inputs, return_logits=False):
        if not self.is_leaf:
            decision = tf.sigmoid((tf.matmul(inputs, self.w) + self.b) * self.inv_temp)
            return decision
        else:
            logits = tf.matmul(inputs, self.phi) + self.phi_b
            if return_logits:
                return logits
            else:
                return tf.nn.softmax(logits)

    def get_loss(self, y_true, y_pred):
        if self.is_leaf:
            return -tf.reduce_mean(y_true * tf.math.log(y_pred + 1e-6))
        else:
            return self.penalty_strength * tf.reduce_mean(tf.square(self.w))

class SoftDecisionTree(tf.keras.Model):
    def __init__(self, max_depth, n_features, n_classes, 
                 penalty_strength, penalty_decay, inv_temp,
                 ema_win_size, learning_rate=3e-4, **kwargs):
        super(SoftDecisionTree, self).__init__(**kwargs)
        self.inv_temp = inv_temp
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - 1/ema_win_size)
        self.nodes = self.create_nodes(0, max_depth, n_features, n_classes, penalty_strength, penalty_decay, inv_temp)
        self.optimizer = Adam(lr=learning_rate)
        self.model = None

    def create_nodes(self, current_depth, max_depth, n_features, n_classes, penalty_strength, penalty_decay, inv_temp):
        if current_depth > max_depth:
            return None
        node = Node(current_depth, max_depth, n_features, n_classes, penalty_strength, penalty_decay, inv_temp)
        left_child = self.create_nodes(current_depth + 1, max_depth, n_features, n_classes, penalty_strength, penalty_decay, inv_temp)
        right_child = self.create_nodes(current_depth + 1, max_depth, n_features, n_classes, penalty_strength, penalty_decay, inv_temp)
        return {'node': node, 'left': left_child, 'right': right_child}

    def call(self, inputs):
        return self.process_node(inputs, self.nodes)
    
    def get_logits(self, inputs):
        return self.process_node(inputs, self.nodes, return_logits=True)

    def process_node(self, inputs, node_dict, return_logits=False):
        node = node_dict['node']
        if node.is_leaf:
            return node(inputs, return_logits=return_logits)
        else:
            decision = node(inputs, return_logits=return_logits)
            left_output = self.process_node(inputs, node_dict['left']) if node_dict['left'] else 0
            right_output = self.process_node(inputs, node_dict['right']) if node_dict['right'] else 0
            return decision * left_output + (1 - decision) * right_output
        

    def get_root_node(self):
        return self.nodes['node']
    
    def print_tree_structure(self, node_dict=None, depth=0, feature_names=None):
        if node_dict is None:
            node_dict = self.nodes
        
        node = node_dict['node']
        
        if not node.is_leaf:
            weight_values = node.w.numpy().flatten()
            feature_index = tf.argmax(tf.abs(weight_values)).numpy()
            feature_name = feature_names[feature_index]
            print(f"{'|   ' * depth}Node at depth {node.depth} - Splitting on feature: {feature_name}")
        else:
            print(f"{'|   ' * depth}Node at depth {node.depth} (Leaf)")
        
        if node_dict['left']:
            self.print_tree_structure(node_dict['left'], depth + 1, feature_names)
        
        if node_dict['right']:
            self.print_tree_structure(node_dict['right'], depth + 1, feature_names)


    def calculate_tree_loss(self, y_true, y_pred):
        # Recursive function to calculate loss for all nodes
        def calculate_node_loss(node_dict):
            node_loss = node_dict['node'].get_loss(y_true, y_pred)
            if 'left' in node_dict and node_dict['left'] is not None:
                node_loss += calculate_node_loss(node_dict['left'])
            if 'right' in node_dict and node_dict['right'] is not None:
                node_loss += calculate_node_loss(node_dict['right'])
            return node_loss

        # Start the recursive loss calculation from the root node
        total_loss = calculate_node_loss(self.nodes)
        return total_loss
    

    def tree_train(self, x_train,y_train, x_val, y_val, batch_size, epochs, distill=False):
        dir_assets = 'assets/'
        dir_distill = 'distilled/'
        dir_non_distill = 'non-distilled/'
        dir_model = dir_assets + (dir_distill if distill else dir_non_distill)
        path_model = tf.train.latest_checkpoint(dir_model)

        if not os.path.exists(dir_model):
            os.makedirs(dir_model)

        if path_model:
            try:
                self.load_weights(path_model)
                # self.compile(optimizer=self.optimizer, loss=self.calculate_tree_loss, metrics=['accuracy'])
                print('Loading trained model from {}.'.format(path_model))
                return
            except ValueError as e:
                print('{} is not a valid checkpoint. Training from scratch.'.format(path_model))
                self.train_new_model(x_train,y_train,x_val, y_val, batch_size, epochs)
        else:
            print('No checkpoint found at {}. Training from scratch.'.format(dir_model))
            self.train_new_model(x_train,y_train,x_val, y_val, batch_size, epochs)
            path_model = dir_model + 'tree-model'

        print('Saving trained model to {}.'.format(path_model))
        self.save_weights(path_model)

    def train_new_model(self, x_train, y_train, x_val, y_val, batch_size, epochs):
        self.compile(optimizer=self.optimizer, loss=self.calculate_tree_loss, metrics=['accuracy'])
        print("Starting training...")
        self.model = self.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs
        )
        print("Training completed.")


