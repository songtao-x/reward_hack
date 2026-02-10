"""
The code is to perform gradient analysis.
1. clustering on two sets of gradient matrix; then calculate accuracy
2. Cosine similarity: get a new gradient, test its cos similarity with two sets of gradient matrix;


"""

import sys
import random
import json
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
import argparse
import torch

import os
from typing import Dict, Any, List
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


from icl.gradient.gradient_h import get_gradients_over_dataset, load_model_and_tokenizer, layer_selection

from utils_ import result_processer

random.seed(224)

SEED=224
MODEL_PATH = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/LLaMA-Factory/saves/qwen3-4b/full/sft/arlsat"   # <- change to your checkpoint




class GradientAnalyzer:
    def __init__(self, args=None):
        self.args = args
        # self.true_gradient_path = args.true_gradient_path
        # self.false_gradient_path = args.false_gradient_path
        # self.new_gradient_path = args.new_gradient_path
    
    def load_gradient(self, INTEST=False):
        self.true_gradients = torch.load(self.args.true_gradient_path)['sketches'].detach().to(dtype=torch.float32, device="cpu")
        self.false_gradients = torch.load(self.args.false_gradient_path)['sketches'].detach().to(dtype=torch.float32, device="cpu")
        
        # self.new_gradients = torch.load(self.new_gradient_path)['sketches']

        if not INTEST and self.args.external_true_gradient_path is not None:
            self.external_true_gradients = torch.load(self.args.external_true_gradient_path)['sketches']
            self.external_false_gradients = torch.load(self.args.external_false_gradient_path)['sketches']
    

    def load_new_gradient(self, true_gradients, false_gradients):
        self.true_gradients = true_gradients
        self.false_gradients = false_gradients

    def gradient_layer_selection(self, model_name, ds):
        selected_layers = layer_selection(ds=ds, model_name=model_name)
        return selected_layers

    def get_gradient(self, ds, model_name, save_path, selected_layers: List, hidden=False, Pi=None):
        """
        Get gradient on file
        Args:
        ds: dataset to get gradient on. The form is List[Dict], keys: "input", "output", "label"
        """

        # Pi = torch.load(f'/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/icl/gradient/data/pi_matrix.pt')
        model, tokenizer = load_model_and_tokenizer(model_name, LORA=True)

        if hidden:
            # get gradient wrt hidden states
            grad, _ = get_gradients_over_dataset(model, tokenizer, ds, LORA=False, HIDDEN=True, Pi=Pi)
        else:
            # get gradient wrt Lora parameters on selected layers
            # selected_layers = layer_selection(ds, Pi=Pi)
            grad, _ = get_gradients_over_dataset(model, tokenizer, ds, LORA=True, layers=selected_layers, HIDDEN=False, Pi=Pi)
        
        torch.save({'sketches': grad}, save_path)
        return grad




    def cos_similarity_analysis(self, new_gradients=None):
        """
        Calulate cos similarity between each new gradient element on true and false gradient sets
        """

        if new_gradients is None:
            new_gradients = self.new_gradients
        
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        true_similarities = []
        false_similarities = []

        for new_grad in new_gradients:
            new_grad = new_grad.unsqueeze(0)
            true_sims = []
            for true_grad in self.true_gradients:
                true_grad = true_grad.unsqueeze(0)
                sim = cos(new_grad, true_grad)
                true_sims.append(sim.item())
            
            false_sims = []
            for false_grad in self.false_gradients:
                false_grad = false_grad.unsqueeze(0)
                sim = cos(new_grad, false_grad)
                false_sims.append(sim.item())
            
            true_similarities.append(np.mean(true_sims))
            false_similarities.append(np.mean(false_sims))
        
        return true_similarities, false_similarities
    
    def verify_cos(self, true, false, SPLIT=False):
        """
        Args:
        true: List[float], true cos results derived from cos_similarity_analysis
        false: List[float], false cos results derived from cos_similarity_analysis
        """

        lens = int(len(true))

        if SPLIT:
            correct = 0
            for t, f in zip(true[:int(lens/2)], false[:int(lens/2)]):
                if t > f:
                    correct += 1
            accuracy = correct / (lens/2)
            print(f'Cos sim results on first half accuracy: {accuracy} = {correct} / {lens/2}')

            correct = 0
            for t, f in zip(true[int(lens/2): ], false[int(lens/2):]):
                if t > f:
                    correct += 1
            accuracy = correct / (lens/2)
            print(f'Cos sim results on second half accuracy: {accuracy} = {correct} / {lens/2}')

            return accuracy
        
        else:
            correct = 0
            for t, f in zip(true, false):
                if t > f:
                    correct += 1
            accuracy = correct / (lens)
            print(f'Ratio of True cos > False cos: {accuracy} = {correct} / {lens}')

            cos_gap = []
            for t, f in zip(true, false):
                cos_gap.append(t - f)

            return accuracy

    def norm_analysis(self):
        """
        Analysis on gradient norms of two sets
        """
        G_true = self.true_gradients.numpy()
        G_false = self.false_gradients.numpy()
        true_norms = np.linalg.norm(G_true, axis=1)
        false_norms = np.linalg.norm(G_false, axis=1)
        print(f'True gradient norms: mean={np.mean(true_norms):.4f}, std={np.std(true_norms):.4f}')
        print(f'False gradient norms: mean={np.mean(false_norms):.4f}, std={np.std(false_norms):.4f}')
        return {"true norm": f'mean={np.mean(true_norms):.4f}, std={np.std(true_norms):.4f}',
                "false norm": f'mean={np.mean(false_norms):.4f}, std={np.std(false_norms):.4f}'}


    def cluster_analysis(self, use_pca=False, use_svd=False, use_t_sne=False, do_plot=False, normalized=True, perp=30):
        """
        Perform clustering analysis on true and false gradient sets
        Default use_pca, use_svd is False
        """

        G_true = self.true_gradients.numpy()   # (N1, D)
        G_false = self.false_gradients.numpy() # (N0, D)

        # G_true:  (N1, D)
        # G_false: (N0, D)
        X = np.concatenate([G_false, G_true], axis=0)
        y = np.concatenate([np.zeros(len(G_false), dtype=int), np.ones(len(G_true), dtype=int)], axis=0)


        # perm = np.random
        # normalize 
        
        if normalized:
            Xn = normalize(X, norm="l2")   # shape (N, D)
        else:
            Xn = X

        # gn = np.linalg.norm(X, axis=1)
        # # try removing magnitude
        # Xn = X / (gn[:,None] + 1e-12)

        print(f'\nXn shape: {Xn.shape}\n')

        # dimension reduction (optional)
        # default is no dimension reduction
        k = min(100, Xn.shape[1])      
        # k = 1024
        if use_pca:
            # PCA 
            Z = PCA(n_components=k, random_state=SEED).fit_transform(Xn)
            # Z = GaussianRandomProjection(n_components=k, random_state=SEED).fit_transform(Xn)
        elif use_svd:
            # SVD
            Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(Xn)
        elif use_t_sne:
            from sklearn.manifold import TSNE
            N  = Xn.shape[0]
            perp = int(min(perp, max(5, (N - 1) // 3)))  # safe for small N
            print(f'perplexity used in t-SNE is {perp}')

            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                init="pca",
                learning_rate="auto",
                random_state=SEED
            )
            Z = tsne.fit_transform(Xn)  # (N, 2)
        else:
            Z = Xn

        # KMeans clustering
        km = KMeans(n_clusters=2, n_init="auto", random_state=SEED)
        c_km = km.fit_predict(Z)
        print(c_km)

        acc_km = self._cluster_accuracy_binary(y, c_km)
        ari_km = adjusted_rand_score(y, c_km)
        nmi_km = normalized_mutual_info_score(y, c_km)

        print(f"KMeans: acc={acc_km:.4f} ARI={ari_km:.4f} NMI={nmi_km:.4f}")

        #  GMM clustering 
        # gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=SEED)

        gmm = GaussianMixture(
            n_components=2,
            covariance_type='diag',
            reg_covar=float(1e-3),
            init_params="random",
            n_init=100,
            max_iter=2000,
            tol=1e-4,
            random_state=SEED
        )
        c_gmm = gmm.fit(Z).predict(Z)

        acc_gmm = self._cluster_accuracy_binary(y, c_gmm)
        ari_gmm = adjusted_rand_score(y, c_gmm)
        nmi_gmm = normalized_mutual_info_score(y, c_gmm)
        print(f"GMM:    acc={acc_gmm:.4f} ARI={ari_gmm:.4f} NMI={nmi_gmm:.4f}")


        # aggolomerative clustering
        ag = AgglomerativeClustering(n_clusters=2, linkage="average", metric="cosine")

        c_ag = ag.fit_predict(Z)
        acc_ag = self._cluster_accuracy_binary(y, c_ag)
        ari_ag = adjusted_rand_score(y, c_ag)
        nmi_ag = normalized_mutual_info_score(y, c_ag)
        print(f"AGG:    acc={acc_ag:.4f} ARI={ari_ag:.4f} NMI={nmi_ag:.4f}")

        # if do plot
        if do_plot:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 3, 1)
            plt.scatter(Z[:, 0], Z[:, 1], c=y, s=18)
            plt.title("t-SNE colored by ground truth y")
            plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
            plt.savefig('tsne_true_label.png')

            plt.subplot(1, 3, 2)
            plt.scatter(Z[:, 0], Z[:, 1], c=c_km, s=18)
            plt.title("t-SNE colored by KMeans clusters")
            plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
            plt.savefig('tsne_kmeans_label.png')

            plt.subplot(1, 3, 3)
            plt.scatter(Z[:, 0], Z[:, 1], c=c_km, s=18)
            plt.title("t-SNE colored by GMM clusters")
            plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
            plt.savefig('tsne_gmm_label.png')

            plt.tight_layout()
            plt.show()
        
        return {"GMM": f"GMM:    acc={acc_gmm:.4f} ARI={ari_gmm:.4f} NMI={nmi_gmm:.4f}",
                "K-means": f"KMeans: acc={acc_km:.4f} ARI={ari_km:.4f} NMI={nmi_km:.4f}",
                "Aggolomerative": f"AGG:    acc={acc_ag:.4f} ARI={ari_ag:.4f} NMI={nmi_ag:.4f}"}


    def _cluster_accuracy_binary(self, y_true, y_pred_cluster):
        # find the larger cluster acc
        acc1 = (y_true == y_pred_cluster).mean()
        acc2 = (y_true == (1 - y_pred_cluster)).mean()
        return max(acc1, acc2)
    

    def _fit_and_eval_kmeans(self, k=128, seed=0):
        G_true = self.true_gradients.numpy()   # (N1, D)
        G_false = self.false_gradients.numpy() # (N0, D)

        X = np.concatenate([G_false, G_true], axis=0)
        y = np.concatenate([np.zeros(len(G_false), dtype=int), np.ones(len(G_true), dtype=int)], axis=0)

        perm = np.random.default_rng(SEED).permutation(X.shape[0])  # set seed if you want reproducible
        X = X[perm]
        y = y[perm]
        
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

        # global preprocessing fit on train only
        Xtr = normalize(Xtr, norm="l2"); Xte = normalize(Xte, norm="l2")
        pca = PCA(n_components=min(k, Xtr.shape[1]), random_state=seed).fit(Xtr)
        Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)

        km = KMeans(n_clusters=2, n_init="auto", random_state=seed).fit(Ztr)
        ctr, cte = km.predict(Ztr), km.predict(Zte)

        # pick mapping using train
        acc_same = (ytr == ctr).mean()
        flip = acc_same < (ytr == (1-ctr)).mean()

        cte_mapped = 1-cte if flip else cte
        acc_te = (yte == cte_mapped).mean()

        print('test acc:', acc_te)
        return acc_te
    
    def g_vendi(self):
        G_true = self.true_gradients[50:, :]
        G_false = self.false_gradients[50:, :]

        true_score = self._g_vendi_score(G_true)
        false_score = self._g_vendi_score(G_false)
        print(f'G-Vendi scores: True set: {true_score:.4f}, False set: {false_score:.4f}')

    # implementation in g vendi paper
    @torch.no_grad()
    def _g_vendi_score(
        self,
        G: torch.Tensor,
        *,
        normalize_rows: bool = True,
        eps: float = 1e-12,
        block_size: int | None = 8192,
    ) -> float:
        """
        Compute G-Vendi score from a gradient matrix G of shape [N, D].

        Paper definition:
        K = (G G^T) / N
        G-Vendi = exp( - sum_i λ_i log λ_i ), where {λ_i} are eigenvalues of K.

        Efficient computation when N >> D:
        Nonzero eigenvalues of (G G^T)/N equal those of (G^T G)/N.
        So we eigendecompose C = (G^T G)/N (shape [D, D]) instead of K (shape [N, N]).

        Args:
        G: [N, D] gradients (projected if you want). Can be fp16/bf16/fp32 on CPU/GPU.
        normalize_rows: if True, row-normalize G to unit norm (as in the paper).
        eps: numerical stability for norms / log.
        block_size: if not None, accumulate C in blocks over N to reduce peak memory.

        Returns:
        G-Vendi score (float).
        """
        if G.ndim != 2:
            raise ValueError(f"G must be 2D [N, D], got {tuple(G.shape)}")

        # Use float32 for accumulation even if G is fp16/bf16
        G = G.to(dtype=torch.float32)

        if normalize_rows:
            G = G / (G.norm(dim=1, keepdim=True) + eps)

        N, D = G.shape
        if N == 0:
            return 0.0

        # C = (G^T G)/N, computed in blocks if desired
        C = torch.zeros((D, D), device=G.device, dtype=torch.float32)
        if block_size is None:
            C = (G.T @ G) / float(N)
        else:
            for s in range(0, N, block_size):
                Gi = G[s : s + block_size]
                C += Gi.T @ Gi
            C /= float(N)

        # Symmetric PSD in theory; use eigvalsh for stability
        evals = torch.linalg.eigvalsh(C)

        # Clamp tiny negatives from numerical error, and renormalize to sum to 1
        evals = torch.clamp(evals, min=0.0)
        total = evals.sum()
        if total > 0:
            evals = evals / total

        # Entropy over positive eigenvalues
        mask = evals > eps
        H = -(evals[mask] * torch.log(evals[mask])).sum()
        return torch.exp(H).item()


    def svm_analysis(self, IN_TEST=True, in_test_ratio=0.5):
        """
        perform SVM classification on two gradient sets
        """
        G_true = self.true_gradients.numpy()   # (N1, D)
        G_false = self.false_gradients.numpy() # (N0, D)

        # Suppose you have:
        # G_true:  (N1, D) 
        # G_false: (N0, D)
        X = np.concatenate([G_false, G_true], axis=0)
        Y = np.concatenate([np.zeros(len(G_false), dtype=int), np.ones(len(G_true), dtype=int)], axis=0)
        X = normalize(X, norm="l2")   # shape (N, D)

        # dimension reduction
        # k = min(1024, X.shape[0])
        # X = TruncatedSVD(n_components=1024, random_state=0).fit_transform(X)
        # X = PCA(n_components=k, random_state=SEED).fit_transform(X)

        # test and train in given subset (split with in_test_ratio)
        if IN_TEST:
            Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=in_test_ratio, random_state=SEED, stratify=Y)
            
            # svm = SVC(kernel='linear', random_state=SEED)
            svm = SVC(kernel='rbf', C=3.0, gamma='scale', random_state=SEED)
            print(f'Using SVM with RBF kernel, C=3.0')
            svm.fit(Xtr, ytr)
            ypred = svm.predict(Xte)
            acc = accuracy_score(yte, ypred)
            # print(f'SVM label: {yte}')
            # print(f'SVM prediction: {ypred}')
            print(f'SVM classification accuracy: {acc:.4f}')

            svm = SVC(kernel='linear', random_state=SEED)
            print(f'Using SVM with linear kernel')
            svm.fit(Xtr, ytr)
            ypred_l = svm.predict(Xte)
            acc_l = accuracy_score(yte, ypred_l)
            # print(f'SVM prediction: {ypred_l}')
            print(f'SVM classification accuracy: {acc_l:.4f}')

            res = {'rbf': {'acc': acc, 'ypred': ypred.tolist()}, 'linear': {'acc': acc_l, 'ypred': ypred_l.tolist()}}
            return res

        else:

            svm = SVC(kernel='linear', random_state=SEED)
            # svm = SVC(kernel='rbf', C=3.0, gamma='scale', random_state=SEED)
            svm.fit(X, Y)

            G_true = self.external_true_gradients.numpy()   # (N1, D)
            G_false = self.external_false_gradients.numpy() # (N0, D)

            # Suppose you have:
            # G_true:  (N1, D)
            # G_false: (N0, D)
            Xte = np.concatenate([G_false, G_true], axis=0)
            Yte = np.concatenate([np.zeros(len(G_false), dtype=int), np.ones(len(G_true), dtype=int)], axis=0)
            Xte = normalize(Xte, norm="l2")   # shape (N, D)

            ypred = svm.predict(Xte)
            acc = accuracy_score(Yte, ypred)
            print(f'SVM classification accuracy on external gradient set: {acc:.4f}')    
            res = {'rbf': {'acc': acc, 'ypred': ypred.tolist()}, 'linear': {'acc': acc_l, 'ypred': ypred_l.tolist()}}
            return res



    def svm_external_test(self, true_set, false_set):
        """
        Perform SVM to test on external datasets
        """

        # train SVM
        G_true = self.true_gradients.numpy()   # (N1, D)
        G_false = self.false_gradients.numpy() # (N0, D)

        # G_true:  (N1, D)
        # G_false: (N0, D)
        X = np.concatenate([G_false, G_true], axis=0)
        Y = np.concatenate([np.zeros(len(G_false), dtype=int), np.ones(len(G_true), dtype=int)], axis=0)
        X = normalize(X, norm="l2")   # shape (N, D)

        # k = min(100, X.shape[1])
        # X = PCA(n_components=k, random_state=SEED).fit_transform(X)

        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=SEED)
        svm.fit(X, Y)

        # test on external sets
        G_true = true_set.numpy()
        G_false = false_set.numpy()
        # G_true:  (N1, D)
        # G_false: (N0, D)
        Xte = np.concatenate([G_false, G_true], axis=0)
        Yte = np.concatenate([np.zeros(len(G_false), dtype=int), np.ones(len(G_true), dtype=int)], axis=0)
        print(Xte.shape)
        Xte = normalize(Xte, norm="l2")   # shape (N, D)
        # Xte = PCA(n_components=k, random_state=SEED).fit_transform(Xte)

        ypred = svm.predict(Xte)
        acc = accuracy_score(Yte, ypred)
        print(f'SVM classification accuracy on external gradient set: {acc:.4f}') 


    def cluster_analysis_on_external(self, true_test, false_test, use_pca=False, use_svd=False, use_t_sne=False, do_plot=False, train_on_external=False, perp=30):
        """
        Perform clustering analysis, train on true and false gradient sets, test on external sets
        Default use_pca, use_svd is False
        """

        if train_on_external:
            G_true = true_test.numpy()   # (N1, D)
            G_false = false_test.numpy() # (N0, D) 
        else: 
            G_true = self.true_gradients.numpy()   # (N1, D)
            G_false = self.false_gradients.numpy() # (N0, D)

        # Suppose you have:
        # G_true:  (N1, D)
        # G_false: (N0, D)
        X = np.concatenate([G_false, G_true], axis=0)
        y = np.concatenate([np.zeros(len(G_false), dtype=int), np.ones(len(G_true), dtype=int)], axis=0)

        perm = np.random.default_rng(SEED).permutation(X.shape[0])  # set seed if you want reproducible
        X = X[perm]
        y = y[perm]
        # 1) (Often important) normalize each sample gradient vector to remove scale
        # Xn = normalize(X, norm="l2")   # shape (N, D)
        # Xn = X

        gn = np.linalg.norm(X, axis=1)
        # try removing magnitude
        Xn = X / (gn[:,None] + 1e-12)

        print(f'\nXn shape: {Xn.shape}\n')

        # 2) Reduce dimension (pick ONE)
        k = min(100, Xn.shape[1])      # tune this

        if use_pca:
            # PCA 
            Z = PCA(n_components=k, random_state=SEED).fit_transform(Xn)
            # Z = GaussianRandomProjection(n_components=k, random_state=SEED).fit_transform(Xn)
        elif use_svd:
            # SVD
            Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(Xn)
        elif use_t_sne:
            from sklearn.manifold import TSNE
            N  = Xn.shape[0]
            perp = int(min(perp, max(5, (N - 1) // 3)))  # safe for small N
            print(f'perplexity used in t-SNE is {perp}')

            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                init="pca",
                learning_rate="auto",
                random_state=SEED
            )
            Z = tsne.fit_transform(Xn)  # (N, 2)
        else:
            Z = Xn

        # KMeans clustering
        km = KMeans(n_clusters=2, n_init="auto", random_state=SEED)
        km.fit(Z)

        # test on external set
        G_true = true_test.numpy()   # (N1, D)
        G_false = false_test.numpy() # (N0, D)

        Xte = np.concatenate([G_false, G_true], axis=0)
        yte = np.concatenate([np.zeros(len(G_false), dtype=int), np.ones(len(G_true), dtype=int)], axis=0)

        gn = np.linalg.norm(Xte, axis=1)
        # try removing magnitude
        Xte = Xte / (gn[:,None] + 1e-12)

        cte_km = km.predict(Xte)
        acc_km = self._cluster_accuracy_binary(yte, cte_km)

        print(f"KMeans on external set: acc={acc_km:.4f}")

        # print(c_km)


        #  GMM clustering 
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=SEED)
        gmm.fit(Z)

        cte_gmm = gmm.predict(Xte)
        acc_gmm = self._cluster_accuracy_binary(yte, cte_gmm)
        print(f"GMM on external set:    acc={acc_gmm:.4f}")


    def cos_sim_on_external(self, true_test, false_test, to_true=True):
        """
        Cosine similarity analysis on external test sets
        Same samples

        Calculate pairwise cos similarity between each test set element and ground true set and false set. 
        Args:
        true_test: list of torch.Tensor
        false_test: list of torch.Tensor
        """

        G_true = true_test
        G_false = false_test

        if to_true:
            comparing_set = self.true_gradients
        else:
            comparing_set = self.false_gradients

        true_sims = []
        for new_grad in G_true:
            new_grad = new_grad.unsqueeze(0)
            sims = []
            for true_grad in comparing_set:
                true_grad = true_grad.unsqueeze(0)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                sim = cos(new_grad, true_grad)
                sims.append(sim.item())
            true_sims.append(np.mean(sims))
        
        false_sims = []
        for new_grad in G_false:
            new_grad = new_grad.unsqueeze(0)
            sims = []
            for false_grad in comparing_set:
                false_grad = false_grad.unsqueeze(0)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                sim = cos(new_grad, false_grad)
                sims.append(sim.item())
            false_sims.append(np.mean(sims))
        
        if true_sims:
            true_sim = np.mean(true_sims)
            false_sim = np.mean(false_sims)
        
        else:
            true_sim = None
            false_sim = None
        # print(f'Cos similarity of true set: {true_sim}')
        # print(f'Cos similarity of false set: {false_sim}')
        return true_sim, false_sim
    # Example:
    # G: torch.Tensor [N, D]
    # score = g_vendi_score(G)
    # print(score)


    def simple_classifer(self, IN_TEST=True, TRAIN=False, classifier_path='simple_classifier.pt'):
        """
        Train a simple MLP classifier on two gradient sets. 
        Split train-test sets randomly at first.
        """
        model = nn.Sequential(
            nn.Linear(self.true_gradients.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        G_true = self.true_gradients.numpy()   # (N1, D)
        G_false = self.false_gradients.numpy() # (N0, D)
        X = np.concatenate([G_false, G_true], axis=0)
        Y = np.concatenate([np.zeros(len(G_false), dtype=int), np.ones(len(G_true), dtype=int)], axis=0)
        X = normalize(X, norm="l2")   # shape (N, D)
        
        if IN_TEST:
            Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.3, random_state=SEED, stratify=Y)
        
        # Training loop
        if TRAIN:
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            epochs = 20
            for epoch in tqdm(range(epochs), desc="Training Classifier: "):
                model.train()
                inputs = torch.tensor(Xtr, dtype=torch.float32)
                labels = torch.tensor(ytr, dtype=torch.long)

                batch_size = 16
                batched_inputs = torch.split(inputs, batch_size)
                batched_labels = torch.split(labels, batch_size)

                for b_inputs, b_labels in zip(batched_inputs, batched_labels):
                    optimizer.zero_grad()
                    outputs = model(b_inputs)
                    loss = criterion(outputs, b_labels)
                    loss.backward()
                    optimizer.step()

                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        else:
            model = torch.load(classifier_path)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(Xte, dtype=torch.float32)
            labels = torch.tensor(yte, dtype=torch.long)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            print(f'Accuracy of the model on the test set: {accuracy:.4f}')

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--true_gradient_path', type=str, default='data/reasonable_gradient_pro')
    parser.add_argument('--false_gradient_path', type=str, default='data/unreasonable_gradient_pro')
    parser.add_argument('--new_gradient_path', type=str, default='data/test_gradient')
    parser.add_argument('--external_true_gradient_path', type=str, default='data/test_true_gradient')
    parser.add_argument('--external_false_gradient_path', type=str, default='data/test_true_gradient')
    args = parser.parse_args()

    # gradient with selected layers based on hidden states cos similarity 
    args.true_gradient_path = 'data_h/true_gradient_hid'
    args.false_gradient_path = 'data_h/false_gradient_hid'

    # gradient wrt to hidden states with selected layers based on hidden states cos similarity 
    # args.true_gradient_path = 'data_h/true_gradient_wrt_hid'
    # args.false_gradient_path = 'data_h/false_gradient_wrt_hid'

    analyzer = GradientAnalyzer(args)
    analyzer.load_gradient(INTEST=True)

    # # Cos sim results on true gradient set
    # true_new = torch.load('data/test_true_gradient')['sketches']
    # true_sims, false_sims = analyzer.cos_similarity_analysis(true_new)
    # accuracy = analyzer.verify_cos(true_sims, false_sims)
    # print(f'True set cos similarity accuracy: {accuracy}')

    #  # Cos sim results on false gradient set
    # false_new = torch.load('data/test_false_gradient')['sketches']
    # true_sims, false_sims = analyzer.cos_similarity_analysis(false_new)
    # accuracy = analyzer.verify_cos(true_sims, false_sims)
    # print(f'False set cos similarity accuracy: {accuracy}')



    # # Clustering analysis
    # analyzer.cluster_analysis(use_t_sne=True, do_plot=True)
    analyzer.cluster_analysis(use_pca=True, do_plot=False)

    analyzer.norm_analysis()

    # SVM classification analysis
    analyzer.svm_analysis(IN_TEST=True, in_test_ratio=0.5)
    # analyzer.svm_analysis(IN_TEST=False)
    input()

    # simple classifier training and evaluating
    analyzer.simple_classifer(TRAIN=True)

    # G-Vendi score
    analyzer.g_vendi()

    # SVM on external test sets
    # Using positive ICL responses
    # true_data = torch.load('data/reasonable_dict_gradient')['sketches']
    # true_gradient = []
    # for true in true_data:
    #     if true:
    #         true_g = true[0]
    #         # true_g = torch.stack(true_g, dim=0)
    #         true_gradient.append(true_g)
    
    # true_gradient = torch.cat(true_gradient, dim=0)
    # torch.save({'sketches': true_gradient}, 'data/pos_icl_true_gradient')
    # print(true_gradient.shape)

    # false_data = torch.load('data/unreasonable_dict_gradient')['sketches']
    # false_gradient = []
    # for false in false_data:
    #     if false:
    #         false_g = false[0]
    #         # false_g = torch.stack(false_g, dim=0)
    #         false_gradient.append(false_g)

    # false_gradient = torch.cat(false_gradient, dim=0)
    # torch.save({'sketches': false_gradient}, 'data/pos_icl_false_gradient')
    # print(false_gradient.shape)

    # true_gradient = torch.load('data/true_resp_gradient')['sketches']
    # false_gradient = torch.load('data/false_resp_gradient')['sketches']

    true_gradient = torch.load('data/test_true_gradient')['sketches']
    false_gradient = torch.load('data/test_false_gradient')['sketches']

    analyzer.svm_external_test(true_gradient, false_gradient)
    analyzer.cluster_analysis_on_external(true_gradient, false_gradient, train_on_external=False)


    # analyzer._fit_and_eval_kmeans(k=128, seed=SEED)



    # cos sim test
    true_gradient_set = torch.load('data/true_set_gradient')['gradient_set']
    false_gradient_set = torch.load('data/false_set_gradient')['gradient_set']

    true_set_sim, false_set_sim = [], []
    to_true = True
    for t_s, f_s in zip(true_gradient_set, false_gradient_set):
        t_sim, f_sim = analyzer.cos_sim_on_external(t_s, f_s, to_true=to_true)
        true_set_sim.append(t_sim)
        false_set_sim.append(f_sim)
    
    # plot 
    y1 = [d for d in true_set_sim if d is not None]
    y2 = [d for d in false_set_sim if d is not None]
    print(len(y1))
    x = range(len(y1))
    higher = [int(y1_>y2_) for y1_, y2_ in zip(y1, y2)]
    plt.plot(x, y1, label='True Set Cos Similarity', color='blue')
    plt.plot(x, y2, label='False Set Cos Similarity', color='red')
    plt.xlabel('Set Index')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarity Analysis to train {to_true}set. Trueset higher: {sum(higher)}/{len(higher)}')
    plt.legend()
    plt.savefig(f'cos_sim_external_sets_to_{to_true}set.png')






