from src.data.scm_generator import SCMGenerator

def verify_generator():
    n_vars = 5
    n_samples = 200
    gen = SCMGenerator(n_vars=n_vars)
    
    tasks = [gen.generate_task(n_samples) for _ in range(100)]
    
    all_validity = []
    for t in tasks:
        all_validity.extend(t['metadata'].ground_truth_validity)
    
    true_rate = sum(all_validity) / len(all_validity)
    print(f"Total tasks checked: {len(tasks)}")
    print(f"Average Validity Rate: {true_rate:.2%}")
    print(f"Expected Rate (with 0.3 corruption per claim): ~70%")

    # check ATE distribution
    ates = [t['metadata'].true_ate for t in tasks]
    print(f"ATE Mean: {sum(ates)/len(ates):.4f}, Std: {np.std(ates):.4f}")

if __name__ == "__main__":
    import numpy as np
    verify_generator()
