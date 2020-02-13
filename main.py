from train_hmm import run
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="code of Corporate Relative Valuation using Heterogeneous Multi-Modal Graph Neural Network")
    parser.add_argument("--data_root", type=str,
                        help="folder of dataset", default="./data")
    parser.add_argument("--adam_alpha", type=float, default=0.5,
                        help="the beta1 argument of Adam optimizer")
    parser.add_argument("--adam_beta", type=float, default=0.99,
                        help="the beta2 argument of Adam optimizer")
    parser.add_argument("--train_ratio", type=int, default=70,
                        help="labeled company node ratio, should be 10, 30, 50 or 70")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50,
                        help="early stop: if validation loss didn't reach new record in patience epoches then stop training")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight_decay in optimizer")
    parser.add_argument("--v1_num", type=int, default=8,
                        help="number of company nodes in neighbourhood")
    parser.add_argument("--v2_num", type=int, default=4,
                        help="number of people nodes in neighbourhood")
    parser.add_argument("--hidden_dim", type=int,
                        default=128, help="hidden layer size")
    parser.add_argument("--v1_edge_attr_dim", type=int, default=1,
                        help="attribute dimension of company-company edges")
    parser.add_argument("--v2_edge_attr_dim", type=int, default=1,
                        help="attribute dimension of company-people edges")
    parser.add_argument("--go_back_p", type=float, default=0.05,
                        help="probability of going back to start node in each step")
    parser.add_argument("--rw_length", type=int, default=300,
                        help="length of random walk")
    parser.add_argument("--vali_ratio", type=float, default=0.1,
                        help="use how many data in validation")
    parser.add_argument("--v1_dim", type=int, default=132,
                        help="company node attribute dimension")
    parser.add_argument("--v2_dim", type=int, default=50,
                        help="people node attribute dimension")
    parser.add_argument("--edge_layer_num", type=int, default=2,
                        help="number of edge aware attention layer")
    parser.add_argument("--edge_type_num", type=int,
                        default=22, help="number of different edge types")
    parser.add_argument("--edge_head_num", type=int, default=4,
                        help="number of heads in edge aware attentino layer")
    parser.add_argument("--edge_emb_dim", type=int,
                        default=32, help="dimension of edge embedding")
    parser.add_argument("--label_num", type=int, default=4,
                        help="number of different labels")
    parser.add_argument("--device", type=str,
                        default="cuda:0", help="cuda or cpu device id")
    parser.add_argument("--epoch_num", type=int,
                        default=1000, help="maximum epoch number")
    parser.add_argument("--triplet_ratio", type=int, default=1,
                        help="we sample triplet_ratio*batch_size triplets in the margin loss")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="margin in margin loss")
    parser.add_argument("--lambda", type=float, default=1e-3,
                        help="weight of margin loss")
    args = parser.parse_args()
    params = vars(args)
    run(params)
