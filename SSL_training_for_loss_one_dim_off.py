# causal_downsampling produces T_enc = T_spec//factor + 1; align to decoder
t_enc = decoder_outputs.shape[1]
if masks.shape[1] < t_enc:
    masks = F.pad(masks, (0, t_enc - masks.shape[1]), value=False)
elif masks.shape[1] > t_enc:
    masks = masks[:, :t_enc]
