import time
import datetime
import logging
import torch
from tools.utils import AverageMeter


def train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes, device, scaler=None):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.to(device), pids.to(device), clothes_ids.to(device), pos_mask.float().to(device)
        
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Update the clothes discriminator
        if config.TRAIN.AMP and scaler is not None:
            with torch.amp.autocast('cuda'):
                features = model(imgs)
                pred_clothes = clothes_classifier(features.detach())
                clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        else:
            features = model(imgs)
            pred_clothes = clothes_classifier(features.detach())
            clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            if config.TRAIN.AMP and scaler is not None:
                scaler.scale(clothes_loss).backward()
                scaler.step(optimizer_cc)
                scaler.update()
            else:
                clothes_loss.backward()
                optimizer_cc.step()

        # Update the backbone
        if config.TRAIN.AMP and scaler is not None:
            with torch.amp.autocast('cuda'):
                features = model(imgs)
                outputs = classifier(features)
                new_pred_clothes = clothes_classifier(features)
                
                # Compute loss
                cla_loss = criterion_cla(outputs, pids)
                pair_loss = criterion_pair(features, pids)
                adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
                
                if epoch >= config.TRAIN.START_EPOCH_ADV:
                    loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
                else:
                    loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss
        else:
            features = model(imgs)
            outputs = classifier(features)
            new_pred_clothes = clothes_classifier(features)
            
            # Compute loss
            cla_loss = criterion_cla(outputs, pids)
            pair_loss = criterion_pair(features, pids)
            adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
            
            if epoch >= config.TRAIN.START_EPOCH_ADV:
                loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
            else:
                loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss
        
        _, preds = torch.max(outputs.data, 1)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)
        
        optimizer.zero_grad()
        if config.TRAIN.AMP and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'PairLoss:{pair_loss.avg:.4f} '
                  'CloLoss:{clo_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   acc=corrects, clo_acc=clothes_corrects))


def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
    criterion_adv, optimizer, trainloader, pid2clothes, device, scaler=None):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.to(device), pids.to(device), clothes_ids.to(device), pos_mask.float().to(device)
        
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Forward with mixed precision if enabled
        if config.TRAIN.AMP and scaler is not None:
            with torch.amp.autocast('cuda'):
                features = model(imgs)
                outputs = classifier(features)
                
                # Compute loss
                cla_loss = criterion_cla(outputs, pids)
                pair_loss = criterion_pair(features, pids)

                if epoch >= config.TRAIN.START_EPOCH_ADV:
                    adv_loss = criterion_adv(features, clothes_ids, pos_mask)
                    loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
                else:
                    loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss
        else:
            features = model(imgs)
            outputs = classifier(features)
            
            # Compute loss
            cla_loss = criterion_cla(outputs, pids)
            pair_loss = criterion_pair(features, pids)

            if epoch >= config.TRAIN.START_EPOCH_ADV:
                adv_loss = criterion_adv(features, clothes_ids, pos_mask)
                loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
            else:
                loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss

        _, preds = torch.max(outputs.data, 1)

        optimizer.zero_grad()
        if config.TRAIN.AMP and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV: 
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                epoch+1, batch_time=batch_time, data_time=data_time, 
                cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                adv_loss=batch_adv_loss, acc=corrects))
