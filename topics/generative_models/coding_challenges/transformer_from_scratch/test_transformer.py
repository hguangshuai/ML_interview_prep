"""
Test suite for Transformer implementation
包含单元测试和集成测试
"""

import torch
import torch.nn as nn
import unittest
import numpy as np
from transformer import (
    PositionalEncoding, MultiHeadAttention, FeedForward, 
    TransformerBlock, Transformer
)

class TestPositionalEncoding(unittest.TestCase):
    """Test positional encoding implementation"""
    
    def setUp(self):
        self.d_model = 512
        self.max_len = 1000
        self.pe = PositionalEncoding(self.d_model, self.max_len)
    
    def test_output_shape(self):
        """Test output shape"""
        batch_size = 32
        seq_len = 100
        x = torch.randn(seq_len, batch_size, self.d_model)
        output = self.pe(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_deterministic(self):
        """Test that positional encoding is deterministic"""
        x = torch.randn(50, 10, self.d_model)
        output1 = self.pe(x)
        output2 = self.pe(x)
        
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths"""
        seq_lengths = [10, 50, 100, 500]
        batch_size = 16
        
        for seq_len in seq_lengths:
            x = torch.randn(seq_len, batch_size, self.d_model)
            output = self.pe(x)
            self.assertEqual(output.shape, (seq_len, batch_size, self.d_model))

class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention implementation"""
    
    def setUp(self):
        self.d_model = 512
        self.num_heads = 8
        self.batch_size = 32
        self.seq_len = 100
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
    
    def test_output_shape(self):
        """Test output shape"""
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output = self.mha(query, key, value)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_attention_weights(self):
        """Test attention mechanism"""
        # Create simple test case
        batch_size = 2
        seq_len = 3
        d_model = 4
        num_heads = 2
        
        mha = MultiHeadAttention(d_model, num_heads)
        
        # Create identical query, key, value
        qkv = torch.randn(batch_size, seq_len, d_model)
        
        output = mha(qkv, qkv, qkv)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
    
    def test_mask_functionality(self):
        """Test attention mask functionality"""
        batch_size = 2
        seq_len = 4
        d_model = 8
        num_heads = 2
        
        mha = MultiHeadAttention(d_model, num_heads)
        
        qkv = torch.randn(batch_size, seq_len, d_model)
        
        # Create mask (mask out last token)
        mask = torch.ones(batch_size, 1, 1, seq_len)
        mask[:, :, :, -1] = 0  # Mask last position
        
        output = mha(qkv, qkv, qkv, mask)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

class TestFeedForward(unittest.TestCase):
    """Test feed-forward network implementation"""
    
    def setUp(self):
        self.d_model = 512
        self.d_ff = 2048
        self.batch_size = 32
        self.seq_len = 100
        self.ff = FeedForward(self.d_model, self.d_ff)
    
    def test_output_shape(self):
        """Test output shape"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.ff(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_nonlinearity(self):
        """Test that ReLU activation is applied"""
        # Create input with negative values
        x = torch.randn(2, 3, self.d_model) - 0.5
        
        output = self.ff(x)
        
        # Check that some values are zero (due to ReLU)
        self.assertTrue(torch.any(output == 0))

class TestTransformerBlock(unittest.TestCase):
    """Test transformer block implementation"""
    
    def setUp(self):
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.batch_size = 32
        self.seq_len = 100
        self.block = TransformerBlock(self.d_model, self.num_heads, self.d_ff)
    
    def test_output_shape(self):
        """Test output shape"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.block(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_residual_connections(self):
        """Test that residual connections work"""
        x = torch.randn(2, 3, self.d_model)
        
        # Test that output is not identical to input (due to transformations)
        output = self.block(x)
        self.assertFalse(torch.allclose(output, x, atol=1e-6))
        
        # Test that output has same shape
        self.assertEqual(output.shape, x.shape)

class TestTransformer(unittest.TestCase):
    """Test complete transformer implementation"""
    
    def setUp(self):
        self.vocab_size = 1000
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        self.batch_size = 32
        self.seq_len = 100
        self.transformer = Transformer(
            self.vocab_size, self.d_model, self.num_heads, 
            self.num_layers, self.d_ff
        )
    
    def test_output_shape(self):
        """Test output shape"""
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = self.transformer(input_ids)
        
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape)
    
    def test_padding_mask(self):
        """Test padding mask creation"""
        # Create input with padding
        input_ids = torch.randint(1, self.vocab_size, (2, 5))
        input_ids[0, -2:] = 0  # Add padding
        input_ids[1, -1:] = 0  # Add padding
        
        mask = self.transformer.create_padding_mask(input_ids)
        
        # Check mask shape
        self.assertEqual(mask.shape, (2, 1, 1, 5))
        
        # Check that padding positions are masked
        self.assertFalse(mask[0, 0, 0, -2:].all())  # Last 2 positions should be False
        self.assertFalse(mask[1, 0, 0, -1:].all())  # Last position should be False
    
    def test_look_ahead_mask(self):
        """Test look-ahead mask creation"""
        size = 5
        mask = self.transformer.create_look_ahead_mask(size)
        
        # Check shape
        self.assertEqual(mask.shape, (size, size))
        
        # Check that it's lower triangular
        for i in range(size):
            for j in range(size):
                if j > i:
                    self.assertFalse(mask[i, j])  # Upper triangle should be False
                else:
                    self.assertTrue(mask[i, j])   # Lower triangle should be True
    
    def test_gradient_flow(self):
        """Test that gradients flow properly"""
        input_ids = torch.randint(0, self.vocab_size, (2, 10))
        output = self.transformer(input_ids)
        
        # Create dummy loss
        target = torch.randint(0, self.vocab_size, (2, 10))
        loss = nn.CrossEntropyLoss()(output.view(-1, self.vocab_size), target.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in self.transformer.parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.all(param.grad == 0))

def run_performance_test():
    """Run performance benchmark"""
    print("🚀 Performance Benchmark")
    print("=" * 50)
    
    import time
    
    # Model parameters
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    batch_size = 32
    seq_len = 128
    
    # Create model
    model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    model.eval()
    
    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)
    
    # Benchmark
    num_runs = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_ids)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = batch_size / avg_time
    
    print(f"📊 Average inference time: {avg_time*1000:.2f} ms")
    print(f"📊 Throughput: {throughput:.1f} samples/sec")
    print(f"📊 Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB" if torch.cuda.is_available() else "📊 Memory usage: CPU mode")

def run_integration_test():
    """Run integration test with training loop"""
    print("🔧 Integration Test - Training Loop")
    print("=" * 50)
    
    # Small model for testing
    vocab_size = 100
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 256
    
    model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 5
    batch_size = 16
    seq_len = 20
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in range(10):  # 10 batches per epoch
            # Create dummy data
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Forward pass
            output = model(input_ids)
            loss = criterion(output.view(-1, vocab_size), target_ids.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / 10
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    print("✅ Integration test completed successfully!")

if __name__ == "__main__":
    # Run unit tests
    print("🧪 Running Unit Tests")
    print("=" * 50)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n")
    
    # Run performance test
    run_performance_test()
    
    print("\n")
    
    # Run integration test
    run_integration_test()
