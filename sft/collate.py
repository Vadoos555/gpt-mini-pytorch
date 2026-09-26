import torch


def create_collate_fn(pad_id):
    def collate_fn(batch):
        
        input_ids_list = []
        target_ids_list = []
        
        max_len = max(len(input_ids) for input_ids, target_ids in batch)
        
        for input_ids, target_ids in batch:
            padding_size = max_len - input_ids.size(0)
            
            if padding_size > 0:
                input_padding = torch.full((padding_size,), pad_id, dtype=torch.long)
                target_padding = torch.full((padding_size,), -100, dtype=torch.long)
                
                input_ids = torch.cat([input_ids, input_padding])
                target_ids = torch.cat([target_ids, target_padding])
            
            input_ids_list.append(input_ids)
            target_ids_list.append(target_ids)
            
        input_ids = torch.stack(input_ids_list)
        target_ids = torch.stack(target_ids_list)
        
        return input_ids, target_ids
    return collate_fn
