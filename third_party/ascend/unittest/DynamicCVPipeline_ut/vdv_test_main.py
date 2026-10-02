import torch
import torch_npu
import time

def call_triton(attention, q, k, v, atten_mask, causal, sm_scale, BM, BN, compile_opt, warm_up_num, time_num):
    stream = torch.npu.current_stream()
    for i in range(warm_up_num):
        tri_out_tmp = attention(q, k, v,atten_mask, causal, sm_scale, BM, BN, compile_opt)
    stream.synchronize()
    
    for i in range(time_num):
        start_time = time.perf_counter()
        tri_out = attention(q, k, v,atten_mask, causal, sm_scale, BM, BN, compile_opt)
        stream.synchronize()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"VDV kernel time: {execution_time} seconds")
    stream.synchronize()
    return tri_out

def call_tnpu(q, k, v, H, atten_mask_golden, sm_scale, sparse_mode, warm_up_num, time_num):
    stream = torch.npu.current_stream()
    for i in range(warm_up_num):
        ref_out_tmp = torch_npu.npu_fusion_attention(
            q, k, v, H,
            padding_mask=None,
            atten_mask=atten_mask_golden,
            scale=sm_scale,
            keep_prob=1.0,
            input_layout='BNSD',
            pre_tockens=65535,
            next_tockens=65535,
            sparse_mode=sparse_mode,
            )[0]
    stream.synchronize()
    for i in range(time_num):
        start_time = time.perf_counter()
        ref_out = torch_npu.npu_fusion_attention(
            q, k, v, H,
            padding_mask=None,
            atten_mask=atten_mask_golden,
            scale=sm_scale,
            keep_prob=1.0,
            input_layout='BNSD',
            pre_tockens=65535,
            next_tockens=65535,
            sparse_mode=sparse_mode,
            )[0]
        stream.synchronize()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"VDV npu_fusion_attention time: {execution_time} seconds")
    stream.synchronize()
    return ref_out

def prepare_data(Z, H, N_CTX, HEAD_DIM, dtype, causal):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()).npu()
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()).npu()
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()).npu()
    sm_scale = 0.5

    atten_mask = None
    atten_mask_golden = None
    sparse_mode= 0
    if causal:
        atten_mask = torch.triu(torch.ones(N_CTX, N_CTX), diagonal=1).bool().npu()
        compressed_len = 2048
        atten_mask_golden = torch.triu(torch.ones(compressed_len, compressed_len), diagonal=1).bool().npu()
        sparse_mode = 2

    return q, k, v, sm_scale, atten_mask, atten_mask_golden, sparse_mode

def check_accuracy(ref_out, tri_out):
    print(ref_out.max().item())

    rtol = 0.0
    atol = 1e-2
    diff_golden_fa = (ref_out - tri_out).abs()
    print(f"diff_golden_fa (Max Diff): {diff_golden_fa.max().item()}")
    if not torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol):
        print("ALLCLOSE failed")
    else:
        print("ALLCLOSE success")
    print("compare success!")


def test_with_ref(unit_under_test, Z, H, N_CTX, HEAD_DIM, causal, dtype,BM ,BN, compile_opt=None, warm_up_num=0, time_num=1):
    print(f"test_with_ref({Z},{H},{N_CTX},{HEAD_DIM}, causal={causal}, dtype={dtype}, BM = {BM},BN = {BN})")
    q, k, v, sm_scale, atten_mask, atten_mask_golden, sparse_mode = prepare_data(Z, H, N_CTX, HEAD_DIM, dtype, causal)
    tri_out = call_triton(unit_under_test, q, k, v, atten_mask, causal, sm_scale, BM, BN, compile_opt, warm_up_num, time_num)
    ref_out = call_tnpu(q, k, v, H, atten_mask_golden, sm_scale, sparse_mode, warm_up_num, time_num)
    check_accuracy(ref_out.cpu(), tri_out.cpu())

def test_unit(unit_under_test, Z, H, N_CTX, HEAD_DIM, causal, dtype,BM ,BN, compile_opt=None, warm_up_num=0, time_num=1):
    print(f"test_unit({Z},{H},{N_CTX},{HEAD_DIM}, causal={causal}, dtype={dtype}, BM = {BM},BN = {BN})")
    q, k, v, sm_scale, atten_mask, atten_mask_golden, sparse_mode = prepare_data(Z, H, N_CTX, HEAD_DIM, dtype, causal)
    tri_out = call_triton(unit_under_test, q, k, v, atten_mask, causal, sm_scale, BM, BN, compile_opt, warm_up_num, time_num)

def test_ref(unit_under_test, Z, H, N_CTX, HEAD_DIM, causal, dtype,BM ,BN, compile_opt=None, warm_up_num=0, time_num=1):
    print(f"test_ref({Z},{H},{N_CTX},{HEAD_DIM}, causal={causal}, dtype={dtype}, BM = {BM},BN = {BN})")
    q, k, v, sm_scale, atten_mask, atten_mask_golden, sparse_mode = prepare_data(Z, H, N_CTX, HEAD_DIM, dtype, causal)
    ref_out = call_tnpu(q, k, v, H, atten_mask_golden, sm_scale, sparse_mode, warm_up_num, time_num)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--uut', 
        type=str, 
        default='both',
        choices=['unit', 'ref', 'both'],        
        #help='What to run? triton unit, torch_npu ref or comparison, (default: %(default), possible values: %(choices))'
    )
    parser.add_argument(
        '--unit-type', 
        type=str, 
        default='par',
        choices=['par', 'orig', 'orig_dag_ssb'],        
        #help='What triton to test? All hw specific in triton (par), original, or with  (default: %(default), possible values: %(choices))'
    )
    parser.add_argument(
        '--dtype', 
        type=str, 
        default='float16',
        choices=['float16', 'bfloat16'],        
        #help='What triton to test? All hw specific in triton (par), original, or with  (default: %(default), possible values: %(choices))'
    )
    parser.add_argument(
        '--tc-params', 
        type=str,
        default=None,
        nargs=7,
    )
    parser.add_argument(
        '--wnum', 
        type=int, 
        default=0,
    )
    parser.add_argument(
        '--tnum', 
        type=int, 
        default=1,
    )
    parser.add_argument('--prof', action='store_true')
    parser.add_argument(
        '--prof-output', 
        type=str,
        default="./prof_output",
    )
    return parser.parse_args()

import triton.runtime.driver as driver
def print_aicore_properties():
    device = torch.npu.current_device()
    print(driver.active.utils.get_device_properties(device))
    
if __name__ == "__main__":
    args = parse_args()
    warm_up_num = args.wnum
    time_num = args.tnum
    uut = args.uut
    unit_type = args.unit_type
    tc_params = args.tc_params
    dtype_str = args.dtype
    is_prof = args.prof
    
    test_op = None
    unit_under_test = None
    
    if uut == 'both':
        test_op = test_with_ref
    elif uut == 'unit':
        test_op = test_unit
    elif uut == 'ref':
        test_op = test_ref
    
    if unit_type == 'par':
        from flash_attention_forward_parallel import attention
        unit_under_test = attention
    elif unit_type == 'orig':
        from FAorigin import attention
        unit_under_test = attention
    elif unit_type == 'orig_dag_ssb':
        from fa_origin_dag_ssb import attention
        unit_under_test = attention
        
    if dtype_str == 'float16':
        dtype = torch.float16
    elif dtype_str == 'bfloat16':
        dtype = torch.bfloat16

    print_aicore_properties()
    
    if is_prof:
        prof_output_dir = args.prof_output
        experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                l2_cache=False,
                data_simplification=False
            )
        
        prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
                ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_output_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config)
    
    if is_prof:
        prof.start()
    if tc_params is not None:
        if tc_params[4] == 'False':
            causal = False
        else:
            causal = True
        test_op(unit_under_test,int(tc_params[0]),int(tc_params[1]),int(tc_params[2]),int(tc_params[3]),causal=causal,dtype=dtype,BM=int(tc_params[5]),BN=int(tc_params[6]),warm_up_num=warm_up_num,time_num=time_num)
    else:
        #test_op(unit_under_test,1,8,8192,128, causal=True, dtype=torch.float16, BM = 32,BN = 32, warm_up_num=warm_up_num, time_num=time_num)
        #test_op(unit_under_test,1,2,2048,128, causal=True, dtype=torch.float16, BM = 128,BN = 128, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,2,2048,128, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,4096,128, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        #test_op(unit_under_test,1,1,64,128, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        #test_op(unit_under_test,1,1,64,128, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,32,32, causal=False, dtype=torch.float16, BM = 32,BN = 32, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,128*8,128, causal=False, dtype=torch.float16, BM = 128,BN = 128, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,64,128, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,128,128, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,64,64, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,64*32,128, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,1024,128, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,128,128, causal=False, dtype=torch.float16, BM = 128,BN = 128, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,4,32,1024,64, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,4,32,4096,64, causal=False, dtype=torch.float16, BM = 64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,4,32,8192,64, causal=False, dtype=torch.float16, BM =64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,4,32,16384,64, causal=False, dtype=torch.float16, BM=64,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,1024,128, causal=False, dtype=torch.float16, BM = 128,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,128,8,1024,128, causal=False, dtype=torch.float16, BM = 128,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,128,8,8192,128, causal=False, dtype=torch.float16, BM = 128,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,128,8,1024,64, causal=False, dtype=torch.float16, BM = 128,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,128,8,8192,64, causal=False, dtype=torch.float16, BM = 128,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,1,1,1024,128, causal=False, dtype=torch.float16, BM = 128,BN = 128, warm_up_num=warm_up_num, time_num=time_num)
        #test_op(unit_under_test,1,1,32,32, causal=True, dtype=torch.float16, BM = 32,BN = 32, warm_up_num=warm_up_num, time_num=time_num)
        #test_op(unit_under_test,1,1,64,32, causal=True, dtype=torch.float16, BM = 32,BN = 32, warm_up_num=warm_up_num, time_num=time_num)
        test_op(unit_under_test,128,8,8192,128, causal=False, dtype=torch.float16, BM = 128,BN = 128, warm_up_num=warm_up_num, time_num=time_num)
        #test_op(unit_under_test,128,8,8192,128, causal=True, dtype=torch.float16, BM = 128,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        #test_op(unit_under_test,128,8,1024,64, causal=True, dtype=torch.float16, BM = 128,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
        #test_op(unit_under_test,128,8,8192,64, causal=True, dtype=torch.float16, BM = 128,BN = 64, warm_up_num=warm_up_num, time_num=time_num)
    if is_prof:
        prof.stop()
        prof.export_chrome_trace(prof_output_dir + "/tnpu_trace.json")
        from torch_npu.profiler.profiler import analyse
        analyse(profiler_path=prof_output_dir)
