import os
import sys

from Saliency_Visualization.Grad_cam_Resnet_50 import GradCamModel
from dataset.dataset_cubs import Dataset_cub

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))

import torch
from torch.utils.data import DataLoader

from dataset.dataset_utils import get_dataset_with_image_and_attributes, \
    get_transform_cub

img_size = 448
seed = 0

# dataset_name: ["mnist", ]
data_root = "/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/data/CUB_200_2011"
json_root = "/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/scripts_data"
dataset_name = "cub"
attribute_file_name = "attributes.npy"

# model_arch: ["Resnet_10", "Resnet_18", Resnet_34, "AlexNet"]
model_arch = "Resnet_50"
pretrained = True
transfer_learning = False
batch_size = 32
lr = 0.001
logs = "/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/Results"
bb_layers_for_concepts = ["layer3"]
last_model_chk_pt_file = "best_val_prune_iteration_11_model_lt.pth.tar"
last_model_mask_file = "lt_mask_non_zero_params_31.5_ite_11.pkl"
num_classes = 200

# cav_flattening_type: ["max_pooled", "avg_pooled", "flattened"]
cav_flattening_type = "flattened"
# cav_flattening_type: "max_pooled"
# cav_flattening_type: "avg_pooled"


continue_pruning = True
initialized_BB_weights = False
# g model configs:
hidden_features = 500
g_lr = 0.001
th = 0
val_after_th = 0
g_epoch = 50
# for max pooled
# g_chkpt: "best_epoch_16.pth.tar"

# for avg pooled
# g_chkpt: "best_epoch_19.pth.tar"

# for flattened
g_chkpt = "best_epoch_42.pth.tar"

# pruning configs
prune_type = "lt"
prune_iterations = 15
start_iter = 0

prune_percent = 10

# epoch for each of the pruned network
end_iter = 60
resample = False
epsilon = 0.0000006

percent_weight_remaining = [
    100.0, 90.0, 81.0, 72.9, 65.6, 59.1, 53.2, 47.9, 43.1, 38.8, 34.9, 31.5,
    28.3, 25.5, 23.0, 20.7, 18.6, 16.8, 15.1, 13.6, 12.3, 11.0, 9.9, 9.0,
    8.1]

n_classes = 200
# These concepts are from LENS paper https://arxiv.org/pdf/2106.06804.pdf
# labels_for_tcav: ["Black_footed_Albatross"]
# concepts_for_tcav: ["has_bill_shape_hooked_seabird", "has_bill_length_about_the_same_as_head",
#                    "has_upper_tail_color_grey", "has_belly_color_white",
#                    "has_wing_shape_roundedwings", "has_bill_color_black"]

# labels_for_tcav: [ "Laysan_Albatross" ]
# concepts_for_tcav: [ "has_crown_color_white",
#                     "has_wing_pattern_solid",
#                     "has_under_tail_color_white" ]

# labels_for_tcav: [ "Groove_billed_Ani" ]
# concepts_for_tcav: [ "has_breast_color_black",
#                     "has_leg_color_black",
#                     "has_bill_shape_allpurpose",
#                     "has_bill_length_about_the_same_as_head",
#                     "has_wing_shape_roundedwings" ]

# labels_for_tcav: [ "Crested_Auklet" ]
# concepts_for_tcav: [ "has_nape_color_black",
#                     "has_eye_color_black",
#                     "has_belly_color_white"]

# labels_for_tcav: [ "Least_Auklet" ]
# concepts_for_tcav: [ "has_breast_color_black",
#                     "has_breast_color_white",
#                     "has_nape_color_white",
#                     "has_size_small_5__9_in"]

# labels_for_tcav: [ "Parakeet_Auklet" ]
# concepts_for_tcav: [ "has_size_medium_9__16_in",
#                     "has_primary_color_white",
#                     "has_leg_color_grey" ]

# labels_for_tcav: [ "Rhinoceros_Auklet" ]
# concepts_for_tcav: [ "has_size_medium_9__16_in",
#                     "has_leg_color_buff"]


concept_names = ['has_bill_shape_dagger', 'has_bill_shape_hooked_seabird',
                 'has_bill_shape_allpurpose', 'has_bill_shape_cone', 'has_wing_color_brown',
                 'has_wing_color_grey', 'has_wing_color_yellow', 'has_wing_color_black',
                 'has_wing_color_white', 'has_wing_color_buff', 'has_upperparts_color_brown',
                 'has_upperparts_color_grey', 'has_upperparts_color_yellow',
                 'has_upperparts_color_black', 'has_upperparts_color_white',
                 'has_upperparts_color_buff', 'has_underparts_color_brown',
                 'has_underparts_color_grey', 'has_underparts_color_yellow',
                 'has_underparts_color_black', 'has_underparts_color_white',
                 'has_underparts_color_buff', 'has_breast_pattern_solid',
                 'has_breast_pattern_striped', 'has_breast_pattern_multicolored',
                 'has_back_color_brown', 'has_back_color_grey', 'has_back_color_yellow',
                 'has_back_color_black', 'has_back_color_white', 'has_back_color_buff',
                 'has_tail_shape_notched_tail', 'has_upper_tail_color_brown',
                 'has_upper_tail_color_grey', 'has_upper_tail_color_black',
                 'has_upper_tail_color_white', 'has_upper_tail_color_buff',
                 'has_head_pattern_plain', 'has_head_pattern_capped',
                 'has_breast_color_brown', 'has_breast_color_grey',
                 'has_breast_color_yellow', 'has_breast_color_black',
                 'has_breast_color_white', 'has_breast_color_buff', 'has_throat_color_grey',
                 'has_throat_color_yellow', 'has_throat_color_black',
                 'has_throat_color_white', 'has_eye_color_black',
                 'has_bill_length_about_the_same_as_head',
                 'has_bill_length_shorter_than_head', 'has_forehead_color_blue',
                 'has_forehead_color_brown', 'has_forehead_color_grey',
                 'has_forehead_color_yellow', 'has_forehead_color_black',
                 'has_forehead_color_white', 'has_forehead_color_red',
                 'has_under_tail_color_brown', 'has_under_tail_color_grey',
                 'has_under_tail_color_yellow', 'has_under_tail_color_black',
                 'has_under_tail_color_white', 'has_under_tail_color_buff',
                 'has_nape_color_blue', 'has_nape_color_brown', 'has_nape_color_grey',
                 'has_nape_color_yellow', 'has_nape_color_black', 'has_nape_color_white',
                 'has_nape_color_buff', 'has_belly_color_grey', 'has_belly_color_yellow',
                 'has_belly_color_black', 'has_belly_color_white', 'has_belly_color_buff',
                 'has_wing_shape_roundedwings', 'has_size_small_5__9_in',
                 'has_size_medium_9__16_in', 'has_size_very_small_3__5_in',
                 'has_shape_perchinglike', 'has_back_pattern_solid',
                 'has_back_pattern_striped', 'has_back_pattern_multicolored',
                 'has_tail_pattern_solid', 'has_tail_pattern_multicolored',
                 'has_belly_pattern_solid', 'has_primary_color_brown',
                 'has_primary_color_grey', 'has_primary_color_yellow',
                 'has_primary_color_black', 'has_primary_color_white',
                 'has_primary_color_buff', 'has_leg_color_grey', 'has_leg_color_black',
                 'has_leg_color_buff', 'has_bill_color_grey', 'has_bill_color_black',
                 'has_crown_color_blue', 'has_crown_color_brown', 'has_crown_color_grey',
                 'has_crown_color_yellow', 'has_crown_color_black', 'has_crown_color_white',
                 'has_wing_pattern_solid', 'has_wing_pattern_striped',
                 'has_wing_pattern_multicolored']

labels = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet',
          'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird',
          'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting',
          'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee',
          'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird',
          'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo',
          'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher',
          'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher',
          'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall',
          'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe',
          'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak',
          'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull',
          'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird',
          'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger',
          'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird',
          'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher',
          'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard',
          'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk',
          'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole',
          'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis',
          'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven',
          'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow',
          'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow',
          'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow',
          'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow',
          'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow',
          'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager',
          'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern',
          'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo',
          'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo',
          'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler',
          'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler',
          'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler',
          'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler',
          'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler',
          'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing',
          'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker',
          'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren',
          'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']

img_dataset, attributes = get_dataset_with_image_and_attributes(
    data_root,
    json_root,
    dataset_name,
    "test",
    attribute_file_name
)

_ite = 3
print(_ite)
checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
chk_pt_file = f"best_val_prune_iteration_{0}_model_lt.pth.tar"
chk_pt_file_name = os.path.join(checkpoint_path, chk_pt_file)
gcmodel = GradCamModel(chk_pt_file_name, dataset_name, n_classes).to("cuda:0")
model_chk_pt = torch.load(chk_pt_file_name)
gcmodel.load_state_dict(model_chk_pt)

transform = get_transform_cub(size=img_size, data_augmentation=True)
dataset = Dataset_cub(img_dataset, attributes, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
data = iter(dataloader)
d = next(data)

print(d[0].size())
img = d[0].to("cuda:0")
out, acts = gcmodel(img)
print(out.size())
print(acts.size())

torch.cuda.empty_cache()
