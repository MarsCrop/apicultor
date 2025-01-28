import nest_asyncio
import asyncio
import concurrent.futures
import multiprocessing
import tracemalloc
from copy import deepcopy
import logging
import warnings
import numpy as np
from functools import reduce

nest_asyncio.apply()

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Start tracing Python memory allocations
tracemalloc.start()

#recommended timeout for fast requests         
#timeout = 100   
#real timeout for the library 
timeout = 8000000     
#ideal timeout for videos         
#timeout = 500 #1000 #2000 #3000        

def is_shareable_data(data):
    """
    Validate provided data is shareable by
    verifying its parallel values are equal.
    If not the data is considered altered and
    can't be shared
    """
    return reduce(lambda x, y: x if np.array_equal(x,y) else None, data) is not None


async def in_executor(loop, executor, func, args, timeout, recursion_size=False, as_singular_matrix = False):
    """
    This method runs the function of a shared worker as a separate
    asynchronious process with a specified timeout in order see more
    directly the output of the function.
    """
    try:             
        print("RECURSION SIZE", recursion_size)
        if recursion_size != False:  # Use None to signal the end of processing
            args.insert(recursion_size[0], recursion_size[1]) 
        print("FUNC", func)
        print("ARGS", args)        
        if as_singular_matrix != False:  # Use None to signal the end of processing
            return await asyncio.wait_for(loop.run_in_executor(executor, func, args), timeout)
        else:  # Use None to signal the end of processing
            return await asyncio.wait_for(loop.run_in_executor(executor, func, *args), timeout)
    except Exception as e: 
        print("Broken request with exception", logger.exception(e))
        print("Broken request")
        print("If data sharing is enabled, this method assumes shared data as first argument of the function and the index as the last argument. Please verify this is correct in your function")
        return "Timeout"
        
async def async_worker(loop, executor, queue, lock, values=None, func=None, index=True, shared = False, continuous = False, recursion_size=False, as_singular_matrix = False, iterroot = False, hybrid = False):
    """
    This method performs as a web worker or as a shared worker
    (with shared parameter being set to True). Users can decide to
    implement functions with an array representing the ith values
    (values) of the updated shared or returned data by the function 
    (func). If shared is set to True, it uses asynchronious locking 
    to update the index of the data in the process. An extra parameter
    called continuous is added in order to apply updates on specific
    indexes of the array by using a scalar index of the array (it can
    be implemented for batch processing).
    Values are always returned by the worker in order to avoid any failure
    condition based on the outcome of the function.
    """
    try:
        args = await queue.get()
        if args is None:  # Use None to signal the end of processing
            queue.task_done()
            return values
        #structure preservation    
        if index is False:  # Use None to signal the end of processing
            args = args
        else:  # Use None to signal the end of processing   
            idx, func_args = args
            if shared == True:
                #Assuming shared data goes first
                #and index in process goes last
                args = [i for i in func_args]
                args.append(idx)
                if iterroot == True:
                    args[1] = args[1][idx]           
            else:
                args = func_args
 
        new_value = await in_executor(loop, executor, func, args, timeout, recursion_size = recursion_size, as_singular_matrix = as_singular_matrix )
        if np.any(new_value == 'Timeout'):
            queue.task_done()
            return values
        # Block to share output with input
        if shared == True:
            async with lock:
                if continuous != False:
                    if idx*continuous >= len(values):
                        queue.task_done()
                        return values
                    else:
                        queue.task_done()
                        if hybrid == False:
                            values[idx*continuous] = new_value
                            queue.task_done()
                            return values
                        else:
                            values[idx*continuous] = new_value[0]
                            queue.task_done()
                            return values, new_value[1:]
                    #print("Shared worker updated data index",idx,"with result",str(values[idx]))
                else:
                    queue.task_done()
                    values[idx] = new_value
                    return values
        else:
            async with lock:
                queue.task_done()
                values = new_value
        #print("Worker updated data index with result",str(values))
        return values
    except Exception as e:
        logger.exception(e)
        return values  

def init_pool_execution_with_queues(max_workers=5):
    """
    This method initializes a queue instance with a lock,
    an event loop and a process pool executor with a 
    specified number of workers for concurrency control.
    """
    queue = asyncio.Queue()
    lock = asyncio.Lock()
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ProcessPoolExecutor()
    executor._max_workers = max_workers
    return queue, lock, loop, executor

async def parallel(values, n_times, args, func=None, index = True, shared=False, fifo = True, lifc = True, continuous = False, recursion_size=False, as_singular_matrix = False, hybrid = False):
    """
    This method performs parallel processing of a function and
    its arguments with support for workers and queueing systems 
    (being configurable with the options fifo and lifc). It also
    includes another to update specific indexes of a function
    by setting the value of the 'continuous' parameter to an index.
    The method uses different parameters for different functionalities:
    if there is a set of values that need to be separately processed or
    updated, the values parameter is set (for example with shared workers
    were a worker can update the index of a value).
    """
    try:
        queue, lock, loop, executor = init_pool_execution_with_queues()
        #print("Running loop with", executor._max_workers, "workers")
        for i in range(len(values)):
            if index == True:
                if shared == False:
                    queue.put_nowait((i, (values[i], *args)))
                else:
                    queue.put_nowait((i, (values, *args)))
            else:
                queue.put_nowait((args))
        tasks = [asyncio.create_task(async_worker(loop, executor, queue, lock, values=values,func=func,index=index,shared=shared, continuous = continuous, recursion_size = recursion_size, as_singular_matrix = as_singular_matrix)) for _ in range(queue.qsize())]
        # Run the queue to be fully processed
        joined_queue = await queue.join()
        # Stop workers with no arguments
        for _prime in tasks:
            queue.put_nowait(None)
        #Multiple low rank adaptation X: gather the whole dataset and perform a tensor product
        values = await asyncio.gather(*tasks)
        if shared == True:
            #Take always the last written copy
            #of the output
            if fifo == True:
                if hybrid == False:
                    assert is_shareable_data(values)
                    values = values[0]
                else:
                    assert is_shareable_data(values[0])
                    read_output = values[1:]
                    values = values[0]
                    return values, read_output
                if lifc == True:
                    fifo = False
            elif lifc == True:
                values = values[-1]
        gathered_output = values
        #print("GATHERED", gathered_output)
        executor.shutdown(wait=False)
        # Close the event loop
        return gathered_output
    except Exception as e:
        logger.exception(e)
        return None
