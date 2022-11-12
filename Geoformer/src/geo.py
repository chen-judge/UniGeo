import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from pathlib import Path
from packaging import version
from tqdm import tqdm
import torch
import logging
from param import parse_args
from geo_data import get_loader
from utils import load_state_dict, LossMeter, set_global_logging_level, AverageMeter
from pprint import pformat

from ManualProgram.eval_equ import Equations
import math

set_global_logging_level(logging.ERROR, ["transformers"])
proj_dir = Path(__file__).resolve().parent.parent


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        from geo_model import VLT5Geo
        model_class = VLT5Geo
        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class, config)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                find_unused_parameters=True
                                )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        self._equ = Equations()
        self.calculation_acc = AverageMeter()
        self.calculation_no_result = AverageMeter()
        self.proving_acc = AverageMeter()
        self.proving_no_result = AverageMeter()

        self.cal_angle = AverageMeter()
        self.cal_length = AverageMeter()
        self.cal_other = AverageMeter()

        self.prove_parallel = AverageMeter()
        self.prove_triangle = AverageMeter()
        self.prove_quadrilateral = AverageMeter()
        self.prove_congruent = AverageMeter()
        self.prove_similarity = AverageMeter()

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
            best_valid = 0.
            best_epoch = 0

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=120)

            epoch_results = {
                'loss': 0.,
            }

            for step_i, batch in enumerate(self.train_loader):
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                update = True
                if self.args.gradient_accumulation_steps > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if update:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    for param in self.model.parameters():
                        param.grad = None
                    global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step}'
                    desc_str += f' | Loss {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

                # Validation
                valid_results = self.evaluate(self.val_loader)

                valid_score = valid_results['BLEU']

                if valid_score > best_valid:
                    best_valid = valid_score
                    best_epoch = epoch
                    self.save("BEST")
                    if epoch >= 20:
                        self.save(f'Epoch{epoch}')

                log_str = ''
                # log_str += pformat(valid_results)
                log_str += "Epoch %d: Valid BLEU %0.4f" % (epoch, valid_score)
                log_str += "\nEpoch %d: Best BLEU %0.4f\n" % (best_epoch, best_valid)

                print(log_str)
                print('save path:', self.args.output)

                self.calculation_acc.reset()
                self.calculation_no_result.reset()
                self.proving_acc.reset()
                self.proving_no_result.reset()

                self.cal_angle.reset()
                self.cal_length.reset()
                self.cal_other.reset()

                self.prove_parallel.reset()
                self.prove_triangle.reset()
                self.prove_quadrilateral.reset()
                self.prove_congruent.reset()
                self.prove_similarity.reset()

            if self.args.distributed:
                dist.barrier()

        # Test Set
        if self.verbose:
            self.save("LAST")
            best_path = os.path.join(self.args.output, 'BEST')
            self.load(best_path)
            self.test()


    def test(self):
        if self.args.distributed:
            dist.barrier()

        if isinstance(self.test_loader, list):
            test_loaders = self.test_loader
        else:
            test_loaders = [self.test_loader]

        for loader in test_loaders:
            split = loader.dataset.source
            test_results = self.evaluate(loader)

            log_str = f'{split} set results\n'
            log_str += pformat(test_results)

            if 'calculation' in split[0]:
                print('Calculation %s Acc %.4f %.4f' % (test_results, self.calculation_acc.get_avg(), self.calculation_no_result.get_avg()))
                print('Subsets: ', self.cal_angle.get_avg(), self.cal_length.get_avg(), self.cal_other.get_avg())
            if 'proving' in split[0]:
                print('Proving %s Acc %.4f %.4f ' % (test_results, self.proving_acc.get_avg(), self.proving_no_result.get_avg()))
                print('Subsets: %.4f %.4f %.4f %.4f %.4f' % (self.prove_parallel.get_avg(), self.prove_triangle.get_avg(),
                                                             self.prove_quadrilateral.get_avg(), self.prove_congruent.get_avg(),
                                                             self.prove_similarity.get_avg()))

    def predict(self, loader, dump_path=None):
        self.model.eval()
        with torch.no_grad():

            predictions = []
            targets = [[]]

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['num_return_sequences'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            for i, batch in enumerate(tqdm(loader, ncols=120, desc=f"Prediction {loader.dataset.source}")):

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                results_with_beams = results['pred'][0:len(results['pred']):self.args.num_beams]
                predictions.extend(results_with_beams)
                targets[0].extend(batch['target_text'])

                self.geo_evaluation(results['pred'], batch)

            assert len(targets) == 1

            results = {
                'predictions': predictions,
                'targets': targets
            }

            if dump_path is not None:
                print('Dumping prediction')
                with open(dump_path, 'w') as f:
                    for i, pred in enumerate(predictions):
                        f.write(pred.lower().strip())
                        if i+1 < len(predictions):
                            f.write('\n')

            return results

    def evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        results = self.predict(loader, dump_path)

        predictions = results['predictions']
        targets = results['targets']
        eval_results = evaluator.evaluate(predictions, targets)
        return eval_results

    def geo_evaluation(self, pred, batch):

        source_nums = batch['source_nums']
        choice_nums = batch['choice_nums']
        label = batch['label']
        problem_form = batch['problem_form']
        target = batch['target_text']
        problem_type = batch['problem_type']

        batch_size = len(source_nums)
        num_beam = self.args.num_beams
        for b in range(batch_size):
            if problem_form[b] == 'calculation':
                choice = self.evaluate_calculation(pred[b*num_beam:(b+1)*num_beam], choice_nums[b], source_nums[b])
                if choice is None:
                    self.calculation_acc.update(0)
                    self.calculation_no_result.update(1.0)
                elif choice == label[b]:
                    self.calculation_acc.update(1.0)
                    self.calculation_no_result.update(0)
                else:
                    self.calculation_acc.update(0)
                    self.calculation_no_result.update(0)

                flag = 1.0 if choice == label[b] else 0
                if problem_type[b] == 'angle':
                    self.cal_angle.update(flag)
                elif problem_type[b] == 'length':
                    self.cal_length.update(flag)
                else:
                    self.cal_other.update(flag)

            else:
                assert problem_form[b] == 'proving'
                success = self.evaluate_proving(pred[b*num_beam:(b+1)*num_beam], target[b])
                if success is None:
                    self.proving_acc.update(0)
                    self.proving_no_result.update(1.0)
                else:
                    self.proving_acc.update(1.0)
                    self.proving_no_result.update(0)

                flag = 0 if success is None else 1.0
                if problem_type[b] == 'parallel':
                    self.prove_parallel.update(flag)
                elif problem_type[b] == 'triangle':
                    self.prove_triangle.update(flag)
                elif problem_type[b] == 'quadrilateral':
                    self.prove_quadrilateral.update(flag)
                elif problem_type[b] == 'congruent':
                    self.prove_congruent.update(flag)
                elif problem_type[b] == 'similarity':
                    self.prove_similarity.update(flag)
                else:
                    assert problem_type[b] == 'proportions'
                    # The proportion problems are also related to triangle
                    self.prove_triangle.update(flag)

    def evaluate_calculation(self, top_k_predictions, choice_nums, source_nums):
        choice = None
        for i in range(self.args.num_beams):
            if choice is not None:
                break
            hypo = top_k_predictions[i].split()
            try:
                res = self._equ.excuate_equation(hypo, source_nums)
            except:
                res = None
            if res is not None and len(res) > 0:
                for j in range(4):
                    if choice_nums[j] is not None and math.fabs(res[-1] - choice_nums[j]) < 0.001:
                        choice = j

        return choice

    def evaluate_proving(self, top_k_predictions, target):
        success = None
        target = target.split()
        for i in range(self.args.num_beams):
            if success is not None:
                break
            hypo = top_k_predictions[i].split()

            if hypo == target:
                success = True

        return success

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    if args.valid_batch_size is not None:
        valid_batch_size = args.valid_batch_size
    else:
        valid_batch_size = args.batch_size

    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
    )

    val_loader = test_loader = None
    if gpu == 0:
        print(f'Building val loader at GPU {gpu}')
        val_loader = get_loader(
            args,
            split=args.valid, mode='val', batch_size=valid_batch_size,
            distributed=False, gpu=args.gpu,
            workers=4,
        )

        print(f'Building test loader at GPU {gpu}')
        if len(args.test.split(',')) == 1:
            test_loader = get_loader(
                args,
                split=args.test, mode='test', batch_size=valid_batch_size,
                distributed=False, gpu=args.gpu,
                workers=4,
            )

        elif len(args.test.split(',')) > 1:
            test_loader = []

            for test_split in args.test.split(','):
                test_loader.append(get_loader(
                    args,
                    split=test_split, mode='test', batch_size=valid_batch_size,
                    distributed=False, gpu=args.gpu,
                    workers=4,
                ))

    trainer = Trainer(args, train_loader, val_loader, test_loader, train=True)

    if not args.test_only:
        trainer.train()
    else:
        trainer.test()



if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
