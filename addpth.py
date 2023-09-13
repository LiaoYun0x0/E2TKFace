import torch

mask = torch.load('Q:\GitHub\PRNetOUT\pth\\rot8SEtopk\\net_050.pth',map_location='cuda:0')

cwm = {"loss.criterion.weight_mask": mask["loss.criterion.weight_mask"]}
cfm = {"loss.criterion.face_mask": mask["loss.criterion.face_mask"]}
mwm = {"loss.metrics.weight_mask": mask["loss.metrics.weight_mask"]}
mfm = {"loss.metrics.face_mask": mask["loss.metrics.face_mask"]}


print(cwm)
# print(cfm)
# print(mwm)
# print(mfm)

# SADRpth = torch.load('D:\GitHub\SADRNet\data\saved_model\SADRNv2\\net_021.pth',map_location='cuda:0')
#
# SADRpth.update(cwm)
# SADRpth.update(cfm)
# SADRpth.update(mwm)
# SADRpth.update(mfm)
#
# torch.save(SADRpth, 'D:\GitHub\SADRNet\data\saved_model\SADRNv2\\net_021getloss.pth', _use_new_zipfile_serialization=False)
