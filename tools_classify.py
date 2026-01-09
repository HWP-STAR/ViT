import time
import torch
from tqdm import tqdm

#基础版本，使用混合精度

def train_basic(model,train_loader,cri,opt,device,batch_size,epochs=5):
    
    #from torch.amp import autocast,GradScaler

    #scaler=GradScaler('cuda')
    model.train()
        
    total_samples=0
    start_time=time.time()

    for epoch in range(epochs):
        epoch_start_time=time.time()    
        total_loss=0
        correct=0
        total=0

        for data,target in train_loader:
            data,target =data.to(device),target.to(device)

            opt.zero_grad()

            #with autocast('cuda'):
            output=model(data)
            loss=cri(output,target)
            #scaler.scale(loss).backward()#反向传播特别
            
            #backward
            loss.backward()
            #scaler.step(opt)
            opt.step()
            #scaler.update()#loss,opt的更新都由scaler决定

            

            total_loss+=loss.item()
            _,predicted=output.max(1)
            total+=target.size(0)
            correct+=predicted.eq(target).sum().item()
            total_samples+=batch_size

        epoch_time=time.time()-epoch_start_time
        put=total_samples/epoch_time
        accuracy=100.*correct/total
        print(f'Epoch{epoch+1}, Loss:{total_loss/len(train_loader):.4f},Acc:{accuracy:.2f} %,Put :{put:.2f} samples/s')

    total_time=time.time()-start_time
    print(f'一共用时：{total_time:.4f} s')
    avg=total_samples*epochs/total_time
    print(f'Avg_put:{avg:.2f} samples/s')


#使用第三方库tdqm，其他一样

def train_basic_2(model,train_loader,cri,opt,device,batch_size,epochs=5):

    model.train()
    
    total_samples=0
    start_time=time.time()

    for epoch in range(epochs):
        epoch_start_time=time.time()    
        total_loss=0
        correct=0
        total=0

        progress_bar=tqdm(train_loader,desc=f'Epoch {epoch+1}',unit="batch")

        for data,target in progress_bar:
            data,target =data.to(device),target.to(device)

            opt.zero_grad()
            output=model(data)
            loss=cri(output,target)
            loss.backward()

            opt.step()

            total_loss+=loss.item()
            _,predicted=output.max(1)
            total+=target.size(0)
            correct+=predicted.eq(target).sum().item()
            total_samples+=batch_size
            progress_bar.set_postfix(loss=loss.item())

        epoch_time=time.time()-epoch_start_time
        put=total_samples/epoch_time
        accuracy=100.*correct/total
        print(f'Epoch{epoch+1}, Loss:{total_loss/len(train_loader):.4f},Acc:{accuracy:.2f} %,Put :{put:.2f} samples/s')

    total_time=time.time()-start_time
    avg=total_samples*epochs/total_time
    print(f'Avg_put:{avg:.2f} samples/s')

def evaluate(model,test_loader,cri,device):
    model.eval()

    total_loss=0.0
    correct=0
    total=0

    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)

            output=model(data)
            loss=cri(output,target)

            total_loss+=loss.item()
            _,predicted=output.max(1)
            total+=target.size(0)
            correct+=predicted.eq(target).sum().item()

    accuracy=100.*correct/total

    print(f'测试集的结果：Loss: {total_loss/len(test_loader):.4f},Acc:{accuracy:.2f}%')

    return accuracy

