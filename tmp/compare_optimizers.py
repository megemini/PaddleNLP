import paddle
import torch
import numpy as np
import matplotlib.pyplot as plt
from adam_mini import Adam_mini
from optimizer import AdamWMini

# Set random seeds for reproducibility
SEED = 1
np.random.seed(SEED)
paddle.seed(SEED)
torch.manual_seed(SEED)
STEPS = 10
VOCAB_SIZE = 100
DTYPE = 'float32'

paddle.set_default_dtype(DTYPE)
torch.set_default_dtype(torch.float16 if DTYPE == 'float16' else torch.float32)

# Set default devices
paddle.set_device('gpu:0')
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleTransformerPaddle(paddle.nn.Layer):
    def __init__(self, dim=2048, n_heads=32):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Embedding layer
        self.embd = paddle.nn.Embedding(VOCAB_SIZE, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))
        
        # Query/Key/Value projections
        self.wq = paddle.nn.Linear(dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))
        self.wk = paddle.nn.Linear(dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))
        self.wv = paddle.nn.Linear(dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))
        
        # Attention projection
        self.wo = paddle.nn.Linear(dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))
        
        # LayerNorm layers
        self.ln1 = paddle.nn.LayerNorm(dim)
        self.ln2 = paddle.nn.LayerNorm(dim)
        
        # MLP layers
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Linear(dim, 4*dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02)),
            paddle.nn.ReLU(),
            paddle.nn.Linear(4*dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))
        )
        
        # Output layer
        self.lm_head = paddle.nn.Linear(dim, VOCAB_SIZE, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))
        
        # Bias parameters
        self.bias = paddle.create_parameter([dim], dtype=DTYPE, default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Embedding
        hidden_states = self.embd(input_ids)  # [batch_size, seq_len, dim]

        # print('1='*10)
        # print(input_ids.shape, input_ids.sum())
        # print(hidden_states.shape, hidden_states.sum())
        
        # Query/Key/Value projections and reshape for multi-head attention
        query = self.wq(hidden_states)  # [batch_size, seq_len, dim]
        key = self.wk(hidden_states)    # [batch_size, seq_len, dim]
        value = self.wv(hidden_states)  # [batch_size, seq_len, dim]
        
        # print('2='*10)
        # print(query.shape, query.sum())
        # print(key.shape, key.sum())
        # print(value.shape, value.sum())

        # Reshape to [batch_size, seq_len, n_heads, head_dim]
        query = query.reshape([batch_size, seq_len, self.n_heads, self.head_dim])
        key = key.reshape([batch_size, seq_len, self.n_heads, self.head_dim])
        value = value.reshape([batch_size, seq_len, self.n_heads, self.head_dim])

        # Transpose to [batch_size, n_heads, seq_len, head_dim]
        query = query.transpose([0, 2, 1, 3])
        key = key.transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = paddle.matmul(query * scale, key.transpose([0, 1, 3, 2]))
        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)
        
        # print('3='*10)
        # print(scale)
        # print(attn_weights.shape, attn_weights.sum())

        # Apply attention to values
        attn_output = paddle.matmul(attn_weights, value)  # [batch_size, n_heads, seq_len, head_dim]
        
        # Reshape back to [batch_size, seq_len, dim]
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, seq_len, self.dim])
        
        # Attention output projection with residual connection and layer norm
        attn_output = self.wo(attn_output)
        hidden_states = self.ln1(hidden_states + attn_output)
        
        # print('4='*10)
        # print(attn_output.shape, attn_output.sum())

        # Feed forward with residual connection and layer norm
        feed_forward = self.mlp(hidden_states)
        hidden_states = self.ln2(hidden_states + feed_forward)
        
        # print('5='*10)
        # print(feed_forward.shape, feed_forward.mean().numpy())

        # Output
        output = self.lm_head(hidden_states + self.bias)

        # print('6='*10)
        # print(output.shape, output.mean().numpy())

        return output


class SimpleTransformerTorch(torch.nn.Module):
    def __init__(self, dim=2048, n_heads=32):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Embedding layer
        self.embd = torch.nn.Embedding(VOCAB_SIZE, dim)
        torch.nn.init.normal_(self.embd.weight, mean=0.0, std=0.02)
        
        # Query/Key/Value projections
        self.wq = torch.nn.Linear(dim, dim)
        self.wk = torch.nn.Linear(dim, dim)
        self.wv = torch.nn.Linear(dim, dim)
        torch.nn.init.normal_(self.wq.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.wk.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.wv.weight, mean=0.0, std=0.02)
        
        # Attention projection
        self.wo = torch.nn.Linear(dim, dim)
        torch.nn.init.normal_(self.wo.weight, mean=0.0, std=0.02)
        
        # LayerNorm layers
        self.ln1 = torch.nn.LayerNorm(dim)
        self.ln2 = torch.nn.LayerNorm(dim)
        
        # MLP layers
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 4*dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4*dim, dim)
        )
        torch.nn.init.normal_(self.mlp[0].weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.02)
    
        # Output layer
        self.lm_head = torch.nn.Linear(dim, VOCAB_SIZE)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        # Bias parameters
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Embedding
        hidden_states = self.embd(input_ids)  # [batch_size, seq_len, dim]
        
        # print('1='*10)
        # print(input_ids.shape, input_ids.sum())
        # print(hidden_states.shape, hidden_states.sum())

        # Query/Key/Value projections and reshape for multi-head attention
        query = self.wq(hidden_states)  # [batch_size, seq_len, dim]
        key = self.wk(hidden_states)    # [batch_size, seq_len, dim]
        value = self.wv(hidden_states)  # [batch_size, seq_len, dim]
        
        # print('2='*10)
        # print(query.shape, query.sum())
        # print(key.shape, key.sum())
        # print(value.shape, value.sum())

        # Reshape to [batch_size, seq_len, n_heads, head_dim]
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose to [batch_size, n_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(query * scale, key.transpose(-2, -1))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # print('3='*10)
        # print(scale)
        # print(attn_weights.shape, attn_weights.sum())

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)  # [batch_size, n_heads, seq_len, head_dim]
        
        # Reshape back to [batch_size, seq_len, dim]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.dim)
        
        # Attention output projection with residual connection and layer norm
        attn_output = self.wo(attn_output)
        hidden_states = self.ln1(hidden_states + attn_output)
        
        # print('4='*10)
        # print(attn_output.shape, attn_output.sum())

        # Feed forward with residual connection and layer norm
        feed_forward = self.mlp(hidden_states)
        hidden_states = self.ln2(hidden_states + feed_forward)
        
        # print('5='*10)
        # print(feed_forward.shape, feed_forward.mean().detach().cpu().numpy())

        # Output
        output = self.lm_head(hidden_states + self.bias)

        # print('6='*10)
        # print(output.shape, output.mean().detach().cpu().numpy())

        return output

def generate_data(batch_size=32, seq_len=64, vocab_size=VOCAB_SIZE):
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    y = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    return x, y

train_data = []
for _ in range(STEPS):
    x_np, y_np = generate_data()
    train_data.append((x_np, y_np))

def train_model(config, steps=STEPS, use_adamw_mini=True):
    losses_both = {}

    model_t = SimpleTransformerTorch(dim=config['dim'], n_heads=config['n_heads']).to(torch_device)
    model_p = SimpleTransformerPaddle(dim=config['dim'], n_heads=config['n_heads'])
    
    print('>>> Torch moduels...')
    for module_name, module in model_t.named_modules(): 
        print(f"Module: {module_name}, Type: {type(module)}")

    print('>>> Paddle sublayers...')
    for module_name, module in model_p.named_sublayers():
        print(f"Module: {module_name}, Type: {type(module)}")

    # Copy parameters from PyTorch model to PaddlePaddle model
    for (name_t, param_t), (name_p, param_p) in zip(model_t.named_parameters(), model_p.named_parameters()):
        # print(f"\nCopying: {name_t} -> {name_p}")
        # print(f"Shapes before: {param_t.shape} -> {param_p.shape}")
        
        param_numpy = param_t.detach().cpu().numpy()
        
        # For linear layers' weights, we need to transpose
        is_weight = any(x in name_t for x in ['weight', 'w_0'])
        is_linear = (
            'mlp' in name_t or  # MLP layer
            'linear' in name_p.lower() or  # Regular linear layers
            any(x in name_t for x in ['wq', 'wk', 'wv', 'wo', 'lm_head'])  # Special layers
        )
        
        # print(f"is_weight: {is_weight}, is_linear: {is_linear}")
        
        if is_weight and is_linear:
            # print(f"Transposing parameter")
            param_numpy = param_numpy.T
            # print(f"Transposed shape: {param_numpy.shape}")
        
        paddle_shape = list(param_p.shape)
        numpy_shape = list(param_numpy.shape)
        if paddle_shape != numpy_shape:
            raise ValueError(f"Shape mismatch after processing: Paddle shape {paddle_shape} != Numpy shape {numpy_shape}")
            
        param_p.set_value(paddle.to_tensor(param_numpy))
        # print(f"Shapes after: {param_t.shape} -> {param_p.shape}")
        # print("-" * 50)

    out_paddle = []
    out_torch = []

    if use_adamw_mini:
        print("Testing AdamWMini (Paddle)...")
    else:
        print("Testing AdamW (Paddle)...")

    # Train Paddle model
    criterion_p = paddle.nn.CrossEntropyLoss()
    model_p.train()
    
    if use_adamw_mini:
        optimizer_p = AdamWMini(
            named_parameters=model_p.named_parameters(),
            learning_rate=config['lr'],
            beta1=config['beta1'],
            beta2=config['beta2'],
            epsilon=config['epsilon'],
            weight_decay=config['weight_decay'],
            dim=config['dim'],
            n_heads=config['n_heads'],
            use_lowprecision_moment=DTYPE == 'float16'
        )
    else:
        optimizer_p = paddle.optimizer.AdamW(
            parameters=model_p.parameters(),
            learning_rate=config['lr'],
            beta1=config['beta1'],
            beta2=config['beta2'],
            epsilon=config['epsilon'],
            weight_decay=config['weight_decay'],
            use_lowprecision_moment=DTYPE == 'float16'
        )

    losses = []
    for step in range(steps):
        x_np, y_np = train_data[step]
        x = paddle.to_tensor(x_np, dtype='int64', place='gpu:0')
        y = paddle.to_tensor(y_np, dtype='int64', place='gpu:0')
        
        out = model_p(x)  # [batch_size, seq_len, vocab_size]
        out = out.reshape([-1, out.shape[-1]])  # [batch_size*seq_len, vocab_size]
        y = y.reshape([-1])  # [batch_size*seq_len]
        loss = criterion_p(out, y)

        # print('o'*20, out.detach().cpu().numpy().reshape(-1)[:10])
        # print('l'*20, out.detach().cpu().numpy().mean(), y.detach().cpu().numpy().sum(), loss.detach().cpu().numpy())

        out_paddle.append(out.detach().cpu().numpy().reshape(-1)[:5])

        model_p.clear_gradients()
        loss.backward()
        optimizer_p.step()
        losses.append(float(loss.numpy()))
        
        if (step+1) % 1 == 0:
            allocated = paddle.device.cuda.memory_allocated()
            reserved = paddle.device.cuda.memory_reserved()
            print(f'step {step+1}, Paddle Loss: {float(loss.numpy()):.4f}')
            print(f'Paddle GPU Memory: Allocated: {allocated/1024**2:.2f}MB, Reserved: {reserved/1024**2:.2f}MB')
            
    losses_both['paddle'] = losses

    if use_adamw_mini:
        print("Testing Adam_mini (PyTorch)...")
    else:
        print("Testing AdamW (PyTorch)...")

    # Train Torch model
    criterion_t = torch.nn.CrossEntropyLoss()
    model_t.train()
    
    if use_adamw_mini:
        optimizer_t = Adam_mini(
            named_parameters = model_t.named_parameters(),
            lr=config['lr'],
            betas= (config['beta1'], config['beta2']),
            eps=config['epsilon'],
            weight_decay=config['weight_decay'],
            dim=config['dim'],
            n_heads=config['n_heads']
        )
    else:
        optimizer_t = torch.optim.AdamW(
            params = model_t.parameters(),
            lr=config['lr'],
            betas= (config['beta1'], config['beta2']),
            eps=config['epsilon'],
            weight_decay=config['weight_decay'],
        )

    
    losses = []
    for step in range(steps):
        x_np, y_np = train_data[step]
        x = torch.tensor(x_np, dtype=torch.long, device=torch_device)
        y = torch.tensor(y_np, dtype=torch.long, device=torch_device)
        
        out = model_t(x)  # [batch_size, seq_len, vocab_size]
        out = out.reshape(-1, out.shape[-1])  # [batch_size*seq_len, vocab_size]
        y = y.reshape(-1)  # [batch_size*seq_len]
        loss = criterion_t(out, y)

        # print('o'*20, out.detach().cpu().numpy().reshape(-1)[:10])
        # print('l'*20, out.detach().cpu().numpy().mean(), y.detach().cpu().numpy().sum(), loss.detach().cpu().numpy())
        out_torch.append(out.detach().cpu().numpy().reshape(-1)[:5])

        optimizer_t.zero_grad()
        loss.backward()
        optimizer_t.step()
        losses.append(float(loss.detach().cpu().numpy()))
        
        if (step+1) % 1 == 0:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            print(f'step {step+1}, Torch Loss: {float(loss.detach().cpu().numpy()):.4f}')
            print(f'Torch GPU Memory: Allocated: {allocated/1024**2:.2f}MB, Reserved: {reserved/1024**2:.2f}MB')
            
    losses_both['torch'] = losses

    print('*'*20)
    print('Compare outputs:')
    for i in range(STEPS):
        print(f'step {i+1}:')
        print('paddle:', out_paddle[i])
        print('torch: ', out_torch[i])

    return losses_both

# 测试配置
configs = [
    {
        'name': 'base_config',
        'lr': 1e-3,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'weight_decay': 0,
        'dim': 2048,
        'n_heads': 32
    },
]

# 运行比较
results = {}
for config in configs:
    print(f"\nTesting config: {config['name']}")
    
    print('='*50)
    print('Test adamw mini...')
    loss = train_model(config)

    results[config['name']] = {
        'Adam_mini': loss['torch'],
        'AdamWMini': loss['paddle']
    }

    print('='*50)
    print('Test original adamw...')
    loss = train_model(config, use_adamw_mini=False)

    results[config['name']] = {
        'Adam_mini': loss['torch'],
        'AdamWMini': loss['paddle']
    }