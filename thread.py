import threading
import queue

# 딥러닝 스레드에서 실행할 함수
def deep_learning_worker(worker_id, job_queue):
    print(f"Deep learning worker {worker_id} started.")
    while True:
        # 큐에서 작업 가져오기 (블로킹 모드)
        job = job_queue.get(block=True)
        if job is None:
            # 작업이 None이면 종료
            break
        print(f"Deep learning worker {worker_id} is processing job: {job}")
        # 딥러닝 작업 수행 (예시로 10초 대기)
        import time
        time.sleep(10)
    print(f"Deep learning worker {worker_id} finished.")

# 딥러닝 스레드들을 저장할 리스트 생성
deep_learning_threads = []

# 딥러닝 작업을 담을 큐 생성
job_queue = queue.Queue()

# 첫번째 딥러닝 스레드 생성 및 시작
t1 = threading.Thread(target=deep_learning_worker, args=(1, job_queue))
t1.start()
deep_learning_threads.append(t1)

# while문에서 스레드 동작
while True:
    # 신호가 오면 딥러닝 작업 추가
    if 조건:
        job_queue.put("new job")
        
    # 딥러닝 스레드들 중 종료된 스레드들은 리스트에서 제거
    for thread in deep_learning_threads:
        if not thread.is_alive():
            deep_learning_threads.remove(thread)
    
    # 모든 딥러닝 스레드들이 종료되면 while문 종료
    if not deep_learning_threads:
        break

# 종료 신호를 보내기 위해 None 추가
for i in range(len(deep_learning_threads)):
    job_queue.put(None)

# 모든 딥러닝 스레드들이 종료될 때까지 대기
for thread in deep_learning_threads:
    thread.join()

print("All threads are done.")