# causal_downsampling produces T_enc = T_spec//factor + 1; align to decoder
t_enc = decoder_outputs.shape[1]
if masks.shape[1] < t_enc:
    masks = F.pad(masks, (0, t_enc - masks.shape[1]), value=False)
elif masks.shape[1] > t_enc:
    masks = masks[:, :t_enc]

-------

# Hey — ran into a shape mismatch training the NEST masked-token-pred model with causal_downsampling: true for streaming. The MLM loss blows up because the mask and the decoder output are off by one in the time dim, e.g. mask [16, 90] but decoder_outputs [16, 91, 8900].
# Root cause: the mask is made at spectrogram res and the loss just floor-divides it by the subsampling factor (T_spec // 8), which equals the encoder length only for symmetric padding. Causal downsampling uses asymmetric padding, so calc_length gives exactly one extra frame (720 → 91 causal vs 90 non-causal). Mask ends up 90, decoder 91, and decoder_outputs[masks] fails on the boolean index.
# Fix I went with — just pad the pooled mask to match the decoder length (extra causal frame padded as False, i.e. unmasked so it adds no loss). In nemo/collections/asr/losses/ssl_losses/mlm.py, right after line 75 (masks = masks.mean(-1) > self.mask_threshold), before line 77 (out_masked_only = decoder_outputs[masks]):
