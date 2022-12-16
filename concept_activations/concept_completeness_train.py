# from collections import OrderedDict
# from datetime import datetime
#
# import torch
# from tqdm import tqdm
#
# import concept_activations.concept_activations_utils as ca_utils
# import lth_pruning.pruning_utils as prune_utils
# import utils
# from concept_activations.g import G
# from run_manager import RunManager
#
#
# def train_concept_to_activation_model(
#         prune_ite,
#         train_loader,
#         val_loader,
#         torch_concept_vector,
#         bb_model,
#         bb_model_meta,
#         model_arch,
#         cav_flattening_type,
#         dataset_name,
#         bb_layer,
#         g_lr,
#         g_model_checkpoint_path,
#         tb_path,
#         epochs,
#         hidden_features,
#         th,
#         val_after_th,
#         num_classes,
#         num_labels,
#         device
# ):
#     final_parameters = OrderedDict(
#         epoch=[epochs],
#         layer=[bb_layer],
#         dataset=[dataset_name],
#         now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')],
#         cav_flattening_type=[cav_flattening_type],
#         prune_iter=[prune_ite]
#     )
#     run_id = utils.get_runs(final_parameters)[0]
#
#     g_model_ip_size, g_model_op_size = ca_utils.get_g_model_ip_op_size(
#         train_loader,
#         device,
#         bb_model,
#         bb_model_meta,
#         torch_concept_vector,
#         bb_layer,
#         cav_flattening_type,
#         dataset_name
#     )
#
#     print(f"===> g_model input size: {g_model_ip_size}")
#     print(f"===> g_model output size: {g_model_op_size}")
#
#     g = G(g_model_ip_size, g_model_op_size, hidden_features).to(device)
#     bb_model_mid, bb_model_tail = ca_utils.dissect_bb_model(model_arch, bb_model)
#     criterion, optimizer, scheduler = get_model_optimization_params(dataset_name, bb_model, g_lr)
#     bb_model.eval()
#
#     run_manager = RunManager(prune_ite, g_model_checkpoint_path, tb_path, train_loader, val_loader)
#     run_manager.begin_run(run_id)
#     for epoch in range(epochs):
#         run_manager.begin_epoch()
#         g.train()
#         with tqdm(total=len(train_loader)) as t:
#             for batch_id, data_tuple in enumerate(train_loader):
#                 images, labels = prune_utils.get_image_target(data_tuple, dataset_name, device)
#                 bs = images.size(0)
#                 _ = bb_model(images)
#
#                 activations = bb_model_meta.model_activations_store[bb_layer]
#                 norm_vc = ca_utils.get_normalized_vc(
#                     activations,
#                     torch_concept_vector,
#                     th,
#                     val_after_th,
#                     cav_flattening_type
#                 )
#                 concept_to_act = g(norm_vc)
#                 concept_to_pred = ca_utils.get_concept_to_pred(
#                     concept_to_act,
#                     bs,
#                     activations,
#                     bb_model_mid,
#                     bb_model_tail
#                 )
#                 optimizer.zero_grad()
#                 train_loss = criterion(concept_to_pred, labels)
#                 train_loss.backward()
#                 optimizer.step()
#
#                 run_manager.track_train_loss(train_loss.item())
#                 run_manager.track_total_train_correct_per_epoch(concept_to_pred, labels, num_classes)
#
#                 t.set_postfix(
#                     epoch='{0}'.format(epoch),
#                     training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
#                 t.update()
#
#         g.eval()
#         out_put_predict_bb = torch.FloatTensor().cuda()
#         out_put_GT = torch.FloatTensor().cuda()
#         with torch.no_grad():
#             with tqdm(total=len(val_loader)) as t:
#                 for batch_id, data_tuple in enumerate(val_loader):
#                     images, labels = prune_utils.get_image_target(data_tuple, dataset_name, device)
#                     bs = images.size(0)
#                     input_to_pred = bb_model(images)
#                     out_put_predict_bb = torch.cat((out_put_predict_bb, input_to_pred), dim=0)
#                     out_put_GT = torch.cat((out_put_GT, labels), dim=0)
#
#                     activations = bb_model_meta.model_activations_store[bb_layer]
#                     norm_vc = ca_utils.get_normalized_vc(
#                         activations,
#                         torch_concept_vector,
#                         th,
#                         val_after_th,
#                         cav_flattening_type
#                     )
#                     concept_to_act = g(norm_vc)
#                     concept_to_pred = ca_utils.get_concept_to_pred(
#                         concept_to_act,
#                         bs,
#                         activations,
#                         bb_model_mid,
#                         bb_model_tail
#                     )
#                     val_loss = criterion(concept_to_pred, labels)
#
#                     run_manager.track_val_loss(val_loss.item())
#                     run_manager.track_total_val_correct_per_epoch(concept_to_pred, labels, num_classes)
#                     t.set_postfix(
#                         epoch='{0}'.format(epoch),
#                         validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
#                     t.update()
#
#         # if scheduler is not None:
#         #     scheduler.step()
#         run_manager.end_epoch(g)
#         bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_classes) / out_put_GT.size(0)
#         epoch_completeness_score = utils.cal_completeness_score(
#             num_labels,
#             run_manager.get_final_val_accuracy(),
#             bb_acc
#         )
#         best_completeness_score = utils.cal_completeness_score(
#             num_labels,
#             run_manager.get_final_best_val_accuracy(),
#             bb_acc
#         )
#         print(f"Epoch: [{epoch + 1}/{epochs}] "
#               f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
#               f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
#               f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} "
#               f"Val_Accuracy: {round(run_manager.get_final_val_accuracy(), 4)} "
#               f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} "
#               f"BB_Val_Accuracy: {round(bb_acc, 4)} "
#               f"Epoch_completeness: {round(epoch_completeness_score, 4)} "
#               f"Best_completeness: {round(best_completeness_score, 4)} "
#               f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")
#
#     run_manager.end_run()
#
#
# def get_model_optimization_params(dataset_name, model, lr):
#     if dataset_name == "mnist":
#         criterion = torch.nn.BCELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#         scheduler = None
#         return criterion, optimizer, scheduler
#     elif dataset_name == "cub":
#         optimizer = torch.optim.SGD(
#             model.parameters(),
#             lr=lr,
#             momentum=0.9,
#             weight_decay=1e-4
#         )
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#         criterion = torch.nn.CrossEntropyLoss()
#         return criterion, optimizer, scheduler


from collections import OrderedDict
from datetime import datetime

import torch
from tqdm import tqdm

import concept_activations.concept_activations_utils as ca_utils
import utils
from concept_activations.g import G
from run_manager import RunManager
import lth_pruning.pruning_utils as prune_utils


def train_concept_to_activation_model(
        prune_ite,
        train_loader,
        val_loader,
        torch_concept_vector,
        bb_model,
        bb_model_meta,
        model_arch,
        cav_flattening_type,
        dataset_name,
        bb_layer,
        g_lr,
        g_model_checkpoint_path,
        tb_path,
        epochs,
        hidden_features,
        th,
        val_after_th,
        num_classes,
        num_labels,
        device
):
    final_parameters = OrderedDict(
        epoch=[epochs],
        layer=[bb_layer],
        dataset=[dataset_name],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')],
        cav_flattening_type=[cav_flattening_type],
        prune_iter=[prune_ite]
    )
    run_id = utils.get_runs(final_parameters)[0]

    g_model_ip_size, g_model_op_size = ca_utils.get_g_model_ip_op_size(
        train_loader,
        device,
        bb_model,
        bb_model_meta,
        torch_concept_vector,
        bb_layer,
        cav_flattening_type,
        dataset_name
    )

    print(f"g_model input size: {g_model_ip_size}")
    print(f"g_model output size: {g_model_op_size}")

    g = G(g_model_ip_size, g_model_op_size, hidden_features).to(device)
    bb_model_mid, bb_model_tail = ca_utils.dissect_bb_model(model_arch, bb_model)
    criterion, optimizer, scheduler = get_model_optimization_params(dataset_name, g, g_lr)

    bb_model.eval()

    run_manager = RunManager(prune_ite, g_model_checkpoint_path, tb_path, train_loader, val_loader)
    run_manager.begin_run(run_id)
    for epoch in range(epochs):
        run_manager.begin_epoch()
        g.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data_tuple in enumerate(train_loader):
                images, labels = prune_utils.get_image_target(data_tuple, dataset_name, device)
                bs = images.size(0)
                _ = bb_model(images)

                activations = bb_model_meta.model_activations_store[bb_layer]
                norm_vc = ca_utils.get_normalized_vc(
                    activations,
                    torch_concept_vector,
                    th,
                    val_after_th,
                    cav_flattening_type
                )
                concept_to_act = g(norm_vc)
                concept_to_pred = ca_utils.get_concept_to_pred(
                    concept_to_act,
                    bs,
                    activations,
                    bb_model_mid,
                    bb_model_tail
                )
                optimizer.zero_grad()
                train_loss = criterion(concept_to_pred, labels)
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(concept_to_pred, labels, num_classes)

                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        g.eval()
        out_put_predict_bb = torch.FloatTensor().cuda()
        out_put_GT = torch.FloatTensor().cuda()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                # for batch_id, (images, labels) in enumerate(val_loader):
                for batch_id, data_tuple in enumerate(val_loader):
                    images, labels = prune_utils.get_image_target(data_tuple, dataset_name, device)
                    bs = images.size(0)
                    input_to_pred = bb_model(images)
                    out_put_predict_bb = torch.cat((out_put_predict_bb, input_to_pred), dim=0)
                    out_put_GT = torch.cat((out_put_GT, labels), dim=0)

                    activations = bb_model_meta.model_activations_store[bb_layer]
                    norm_vc = ca_utils.get_normalized_vc(
                        activations,
                        torch_concept_vector,
                        th,
                        val_after_th,
                        cav_flattening_type
                    )
                    concept_to_act = g(norm_vc)
                    concept_to_pred = ca_utils.get_concept_to_pred(
                        concept_to_act,
                        bs,
                        activations,
                        bb_model_mid,
                        bb_model_tail
                    )
                    val_loss = criterion(concept_to_pred, labels)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(concept_to_pred, labels, num_classes)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(g)
        bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_classes) / out_put_GT.size(0)
        epoch_completeness_score = utils.cal_completeness_score(
            num_labels,
            run_manager.get_final_val_accuracy(),
            bb_acc
        )
        best_completeness_score = utils.cal_completeness_score(
            num_labels,
            run_manager.get_final_best_val_accuracy(),
            bb_acc
        )
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} "
              f"Val_Accuracy: {round(run_manager.get_final_val_accuracy(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} "
              f"BB Val_Accuracy: {round(bb_acc, 4)} "
              f"Epoch_completeness: {round(epoch_completeness_score, 4)} "
              f"Best_completeness: {round(best_completeness_score, 4)} "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")

    run_manager.end_run()


def get_model_optimization_params(dataset_name, network, lr):
    if dataset_name == "mnist":
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        scheduler = None
        return criterion, optimizer, scheduler
    elif dataset_name == "cub":
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(
        #     network.parameters(),
        #     lr=lr,
        #     momentum=0.9,
        #     weight_decay=1e-4
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        return criterion, optimizer, scheduler
