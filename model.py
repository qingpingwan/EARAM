import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class multimodal_attention(nn.Module):
    """
    dot-product attention mechanism
    """

    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.matmul(q, k.transpose(-2, -1))
        # print('attention.shape:{}'.format(attention.shape))
        if scale:
            attention = attention * scale

        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
            
        attention = self.softmax(attention)
        # print('attention.shftmax:{}'.format(attention))
        attention = self.dropout(attention)
        v_result = torch.matmul(attention, v)
        # print('attn_final.shape:{}'.format(attention.shape))

        return v_result





class CrossAttention(nn.Module):
    """
    Multi-Head Cross Attention mechanism
    """

    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(CrossAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query

        # Linear projection
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # Split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Scaled dot product attention
        scale = (self.dim_per_head) ** -0.5
        attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        # Final linear projection
        output = self.linear_final(attention)

        # Dropout
        output = self.dropout(output)

        # Add residual and norm layer
        output = self.layer_norm(residual + output)

        return output

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(MultiHeadCrossAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.cross_attention = CrossAttention(model_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, attn_mask=None):
        # Cross attention from x1 to x2
        cross_attn_output_1 = self.cross_attention(x1, x2, x2, attn_mask)
        # Cross attention from x2 to x1
        cross_attn_output_2 = self.cross_attention(x2, x1, x1, attn_mask)

        # Combine the outputs
        output_1 = self.layer_norm(x1 + cross_attn_output_1)
        output_2 = self.layer_norm(x2 + cross_attn_output_2)

        return output_1, output_2

# # Example usage
# batch_1, len_1, dim = 2, 10, 768
# batch_2, len_2, dim = 2, 15, 768

# x1 = torch.randn(batch_1, len_1, dim)
# x2 = torch.randn(batch_2, len_2, dim)

# layer = MultiHeadCrossAttention(model_dim=768, num_heads=8, dropout=0.5)
# output_1, output_2 = layer(x1, x2)


# print("output_1 shape:", output_1.size())  # Expected: [batch_1, len_1, 768]
# print("output_2 shape:", output_2.size())  # Expected: [batch_2, len_2, 768]


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention mechanism
    """

    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(MultiHeadSelfAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, attn_mask=None):
        residual = x

        # Linear projection
        query = self.linear_q(x)
        key = self.linear_k(x)
        value = self.linear_v(x)

        # Split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Scaled dot product attention
        scale = (self.dim_per_head) ** -0.5
        attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        # Final linear projection
        output = self.linear_final(attention)

        # Dropout
        output = self.dropout(output)

        # Add residual and norm layer
        output = self.layer_norm(residual + output)

        return output

# # Example usage
# batch_size = 2
# seq_len = 10
# model_dim = 768

# x = torch.randn(batch_size, seq_len, model_dim)

# self_attention = MultiHeadSelfAttention(model_dim=model_dim, num_heads=8, dropout=0.5)
# output = self_attention(x)

# print("output shape:", output.size())  # Expected: [batch_size, seq_len, model_dim]


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = model_dim // num_heads
        
        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(model_dim, model_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim_per_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)
        output = self.linear_out(context)
        
        return output




# class CoAttention_2(nn.Module):
#     def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
#         super(CoAttention, self).__init__()
#         self.attention1 = MultiHeadAttention(model_dim, num_heads, dropout)
#         self.attention2 = MultiHeadAttention(model_dim, num_heads, dropout)
#         self.linear_out = nn.Linear(2 * model_dim, model_dim)
#         self.layer_norm1 = nn.LayerNorm(model_dim)
#         self.layer_norm2 = nn.LayerNorm(model_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x1, x2, mask1=None, mask2=None):
#         attn_output1 = self.layer_norm1(x1 + self.attention1(x1, x2, x2, mask2))
#         attn_output2 = self.layer_norm2(x2 + self.attention2(x2, x1, x1, mask1))
#         # print(attn_output1.size())
#         # print(attn_output2.size())
        
        
#         pooled1 = attn_output1.max(dim=1)[0]
#         pooled2 = attn_output2.max(dim=1)[0]
        
#         # print(pooled1.size())
#         # print(pooled2.size())
        
#         combined = torch.cat([pooled1, pooled2], dim=-1)
#         output = self.dropout(self.linear_out(combined))
        
#         return output

# class CoAttention_3(nn.Module):
#     def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
#         super(CoAttention_3, self).__init__()
#         self.model_dim = model_dim
#         self.num_heads = num_heads
#         self.head_dim = model_dim // num_heads
        
#         self.query_weight = nn.Parameter(torch.randn(model_dim, model_dim))
#         self.key_weight = nn.Parameter(torch.randn(model_dim, model_dim))
#         self.value_weight = nn.Parameter(torch.randn(model_dim, model_dim))
        
#         self.query_weight2 = nn.Parameter(torch.randn(model_dim, model_dim))
#         self.key_weight2 = nn.Parameter(torch.randn(model_dim, model_dim))
#         self.value_weight2 = nn.Parameter(torch.randn(model_dim, model_dim))
        
#         self.out_weight = nn.Parameter(torch.randn(2 * model_dim, model_dim))
        
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm1 = nn.LayerNorm(model_dim)
#         self.layer_norm2 = nn.LayerNorm(model_dim)
        
#     def split_heads(self, x, batch_size):

#         return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
#     def scaled_dot_product_attention(self, query, key, value, mask=None):

#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
        
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_output = torch.matmul(attn_weights, value)
#         return attn_output
    
#     def multi_head_attention(self, x1, x2, q_weight, k_weight, v_weight, mask=None):
#         batch_size = x1.size(0)
        

#         query = torch.matmul(x1, q_weight)
#         key = torch.matmul(x2, k_weight)
#         value = torch.matmul(x2, v_weight)
        

#         query = self.split_heads(query, batch_size)
#         key = self.split_heads(key, batch_size)
#         value = self.split_heads(value, batch_size)
        

#         attn_output = self.scaled_dot_product_attention(query, key, value, mask)
        

#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
#         return attn_output
    
#     def forward(self, x1, x2, mask1=None, mask2=None):

#         attn_output1 = self.layer_norm1(x1 + self.multi_head_attention(x1, x2, self.query_weight, self.key_weight, self.value_weight, mask2))
#         attn_output2 = self.layer_norm2(x2 + self.multi_head_attention(x2, x1, self.query_weight2, self.key_weight2, self.value_weight2, mask1))
        

#         pooled1 = attn_output1.max(dim=1)[0]
#         pooled2 = attn_output2.max(dim=1)[0]
        

#         combined = torch.cat([pooled1, pooled2], dim=-1)
#         output = self.dropout(torch.matmul(combined, self.out_weight))
        
#         return output

# def test_CoAttention_3():


#     model_dim = 768
#     num_heads = 8
#     dropout = 0.5
#     coattention = CoAttention_3(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
    

#     batch_size = 4
#     seq_len1 = 10  
#     seq_len2 = 12  
    

#     x1 = torch.randn(batch_size, seq_len1, model_dim)
#     x2 = torch.randn(batch_size, seq_len2, model_dim)
    

#     mask1 = torch.randint(0, 2, (batch_size, 1, 1, seq_len1)).to(torch.float32)
#     mask2 = torch.randint(0, 2, (batch_size, 1, 1, seq_len2)).to(torch.float32)
    

#     print("Testing with random input tensors:")
#     output = coattention(x1, x2, mask1, mask2)
#     print(f"Output shape: {output.shape}")
    
#     print("\nTesting without masks:")
#     output = coattention(x1, x2)
#     print(f"Output shape: {output.shape}")
#     assert output.shape == (batch_size, model_dim), "dim should: (batch_size, model_dim)"
    

#     seq_len1_alt = 15
#     seq_len2_alt = 8
#     x1_alt = torch.randn(batch_size, seq_len1_alt, model_dim)
#     x2_alt = torch.randn(batch_size, seq_len2_alt, model_dim)
    
#     print("\nTesting with different sequence lengths:")
#     output = coattention(x1_alt, x2_alt)
#     print(f"Output shape: {output.shape}")
#     assert output.shape == (batch_size, model_dim), "dim should: (batch_size, model_dim)"
    

#     min_seq_len1 = 1
#     min_seq_len2 = 1
#     x1_min = torch.randn(batch_size, min_seq_len1, model_dim)
#     x2_min = torch.randn(batch_size, min_seq_len2, model_dim)
    
#     print("\nTesting with minimum sequence lengths (1):")
#     output = coattention(x1_min, x2_min)
#     print(f"Output shape: {output.shape}")
#     assert output.shape == (batch_size, model_dim), "dim should: (batch_size, model_dim)"
    

# # Example usage
# batch_size, len_1, len_2, dim = 2, 10, 15, 768

# x1 = torch.randn(batch_size, len_1, dim)
# x2 = torch.randn(batch_size, len_2, dim)

# model = CoAttention(model_dim=dim, num_heads=8, dropout=0.5)
# output = model(x1, x2)

# print("output shape:", output.size())  # Expected: [batch, 768]




def adaptive_resize(tensor, target_len):
    return F.adaptive_avg_pool2d(tensor.transpose(1, 2), (target_len, tensor.size(2)))

class CoAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(CoAttention, self).__init__()
        self.attention1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention2 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.linear_out = nn.Linear(2 * model_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2, mask1=None, mask2=None):
        attn_output1 = self.attention1(x1, x2, x2, mask2)
        attn_output2 = self.attention2(x2, x1, x1, mask1)
        
        combined_1 = torch.cat([attn_output1.mean(dim=1), attn_output2.mean(dim=1)], dim=-1)
        output_1 = self.dropout(self.linear_out(combined_1))
        output_1 = self.layer_norm(output_1)
        
        attn_output2_new = adaptive_resize(attn_output2, x1.size(1))

        combined_2 = torch.cat([attn_output1, attn_output2_new], dim=-1)
        
        output_2 = self.dropout(self.linear_out(combined_2))
        output_2 = self.layer_norm(output_2)
        
        return output_1, output_2
    
    
class PositionalWiseFeedForward(nn.Module):
    """
    Fully-connected network
    """

    def __init__(self, model_dim=768, ffn_dim=2048, dropout=0.5):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x

        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        output = x
        return output
    
    
    
    


class MLP(nn.Module):
    def __init__(self, in_features, out_features =2, hidden_size=256, dropout=0.5):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class simple_mlp(nn.Module):
    def __init__(self,in_features,out_features =2, hidden_size=256, dropout= 0.5):
        super(simple_mlp,self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=-1)
        return x
    
    
    
    
class VLR(nn.Module):
    def __init__(self, dim=512):
        super(VLR, self).__init__()
        self.cross_layer_1 = MultiHeadCrossAttention(model_dim=dim, num_heads=8, dropout=0.5)
        self.co_layer_1 = CoAttention(model_dim=dim, num_heads=8, dropout=0.5)
        self.cross_layer_2 = MultiHeadCrossAttention(model_dim=dim, num_heads=8, dropout=0.5)
        self.co_layer_2 = CoAttention(model_dim=dim, num_heads=8, dropout=0.5)
        
        # Define a learnable parameter for the weighted sum
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.mlp = MLP(in_features=dim, out_features=2)
        self.sim_mlp_1 = simple_mlp(in_features=dim)
        self.sim_mlp_2 = simple_mlp(in_features=dim)
        

    def forward(self, x1, x2, x3, x4):
        x1, x2 = self.cross_layer_1(x1, x2)
        
        fusion_1,fusion_high_dim = self.co_layer_1(x1, x2)
        
        x3,_ = self.cross_layer_2(x3, fusion_high_dim)
        
        x4,_ = self.cross_layer_2(x4, fusion_high_dim)
        
        fusion_2,_ = self.co_layer_2(x3, x4)
        
        # Weighted sum of fusion_1 and fusion_2

        output = self.alpha * fusion_1 + (1 - self.alpha) * fusion_2
        
        output = self.mlp(output)
        output = F.softmax(output, dim=-1)
        
        output_1 = self.sim_mlp_1(x3)
        output_2 = self.sim_mlp_2(x4)
        
        res = {
            "label_1":output,
            "label_2":output_1,
            "label_3":output_2            
        }
        return res
