#include <iostream>
#include <cstdlib>
#include <map>
#include <vector>
#include <algorithm>
using namespace std;

#define MAX_TREE_HT 33

struct MinHeapNode{
    int data;
    unsigned freq;
    struct MinHeapNode *left, *right;
};

struct MinHeap{
    unsigned size;
    unsigned capacity;
    struct MinHeapNode** array;
};
// A utility function allocate a new
// min heap node with given character
// and frequency of the character
struct MinHeapNode* newNode(int data, unsigned freq){
    struct MinHeapNode* temp
        = (struct MinHeapNode*)malloc
(sizeof(struct MinHeapNode));
    temp->left = temp->right = NULL;
    temp->data = data;
    temp->freq = freq;
    return temp;
}

// A utility function to create
// a min heap of given capacity
struct MinHeap* createMinHeap(unsigned capacity){
    struct MinHeap* minHeap
        = (struct MinHeap*)malloc(sizeof(struct MinHeap));
    // current size is 0
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array
        = (struct MinHeapNode**)malloc(minHeap->
capacity * sizeof(struct MinHeapNode*));
    return minHeap;
}

// A utility function to
// swap two min heap nodes
void swapMinHeapNode(struct MinHeapNode** a,
                    struct MinHeapNode** b){
    struct MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}

// The standard minHeapify function.
void minHeapify(struct MinHeap* minHeap, int idx){ 
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    if (left < minHeap->size && minHeap->array[left]->
freq < minHeap->array[smallest]->freq)
        smallest = left;
    if (right < minHeap->size && minHeap->array[right]->
freq < minHeap->array[smallest]->freq)
        smallest = right;
    if (smallest != idx) {
        swapMinHeapNode(&minHeap->array[smallest],
                        &minHeap->array[idx]);
        minHeapify(minHeap, smallest);
    }
}

// A utility function to check
// if size of heap is 1 or not
int isSizeOne(struct MinHeap* minHeap){
    return (minHeap->size == 1);
}
 
// A standard function to extract
// minimum value node from heap
struct MinHeapNode* extractMin(struct MinHeap* minHeap){ 
    struct MinHeapNode* temp = minHeap->array[0];
    minHeap->array[0]
        = minHeap->array[minHeap->size - 1];
    --minHeap->size;
    minHeapify(minHeap, 0);
    return temp;
}

// A utility function to insert
// a new node to Min Heap
void insertMinHeap(struct MinHeap* minHeap,
                struct MinHeapNode* minHeapNode){ 
    ++minHeap->size;
    int i = minHeap->size - 1;
 
    while (i && minHeapNode->freq < minHeap->array[(i - 1) / 2]->freq) {
        minHeap->array[i] = minHeap->array[(i - 1) / 2];
        i = (i - 1) / 2;
    }
 
    minHeap->array[i] = minHeapNode;
}

void buildMinHeap(struct MinHeap* minHeap){ 
    int n = minHeap->size - 1;
    int i;
    for (i = (n - 1) / 2; i >= 0; --i)
        minHeapify(minHeap, i);
}

// Utility function to check if this node is leaf
int isLeaf(struct MinHeapNode* root){ 
    return !(root->left) && !(root->right);
}

// Creates a min heap of capacity
// equal to size and inserts all character of
// data[] in min heap. Initially size of
// min heap is equal to capacity
struct MinHeap* createAndBuildMinHeap(map<int, double> P, int size){ 
    struct MinHeap* minHeap = createMinHeap(size); 
    int i=0;
    for(map<int, double>::iterator it = P.begin(); it != P.end(); it++){
        minHeap->array[i] = newNode(it->first, it->second);
        i++;
    }
    minHeap->size = size;
    buildMinHeap(minHeap);
    return minHeap;
}

// The main function that builds Huffman tree
struct MinHeapNode* buildHuffmanTree(map<int, double> P, int size){
    struct MinHeapNode *left, *right, *top;
    // Step 1: Create a min heap of capacity
    // equal to size. Initially, there are
    // modes equal to size.
    struct MinHeap* minHeap = createAndBuildMinHeap(P, size); 
    // Iterate while size of heap doesn't become 1
    while (!isSizeOne(minHeap)){ 
        // Step 2: Extract the two minimum
        // freq items from min heap
        left = extractMin(minHeap);
        right = extractMin(minHeap);
        // Step 3: Create a new internal
        // node with frequency equal to the
        // sum of the two nodes frequencies.
        // Make the two extracted node as
        // left and right children of this new node.
        // Add this node to the min heap
        // '$' is a special value for internal nodes, not used
        top = newNode(-1, left->freq + right->freq);
        top->left = left;
        top->right = right;
        insertMinHeap(minHeap, top);
    }
    // Step 4: The remaining node is the
    // root node and the tree is complete.
    return extractMin(minHeap);
}

// Prints huffman codes from the root of Huffman Tree.
// It uses arr[] to store codes
void printCodes(struct MinHeapNode* root, int arr[], int top, map<int, int>& huffman_table_length){
    // Assign 0 to left edge and recur
    if (root->left){
        arr[top] = 0;
        printCodes(root->left, arr, top + 1, huffman_table_length);
    }

    // Assign 1 to right edge and recur
    if (root->right){
        arr[top] = 1;
        printCodes(root->right, arr, top + 1, huffman_table_length);
    }
    // If this is a leaf node, then
    // it contains one of the input
    // characters, print the character
    // and its code from arr[]
    if (isLeaf(root)){
        huffman_table_length[root->data]=top;
        // cout<< root->data <<": ";
        // printArr(arr, top);
    }
}

// The main function that builds a
// Huffman Tree and print codes by traversing
// the built Huffman Tree
void HuffmanCodes(map<int, double> P, int size, map<int, int>& huffman_table_length){
    // Construct Huffman Tree
    struct MinHeapNode* root
        = buildHuffmanTree(P, size);
    // Print Huffman codes using
    // the Huffman tree built above
    int arr[MAX_TREE_HT], top = 0;
 
    printCodes(root, arr, top, huffman_table_length);
}

void adjustBitLengthTo16Bits(int BITS[33]){
    // for(int x=0;x<32;x++){
    //     cout<<x<<"-->"<<BITS[x]<<endl;
    // }
    int i=32,j=0;
    while(1){
        if(BITS[i]>0){
            j=i-1;
            j--;
            while(BITS[j]<=0){j--;}
            BITS[i]=BITS[i]-2;
            BITS[i-1]=BITS[i-1]+1;
            BITS[j+1]=BITS[j+1]+2;
            BITS[j]=BITS[j]-1;
            continue;
        }
        else{
            i--;
            if(i!=16){continue;}
            while(BITS[i]==0){i--;}
            //BITS[i]--;
            return;
        }
    }
}

void numOfCodesOfEachSize(map<int, int> huffman_table_length, int BITS[33]){
    int len;
    for(int i=0;i<257;i++){
        len = huffman_table_length[i];
        if(len!=0){
            BITS[len] += 1;
        }
    }
}

void HuffmanSize(int BITS[33], map<int, int> & huffman_size, 
                 std::multimap<double, int> SortedP, int size){
    vector<int> sizelst;
    for(int i=0; i<17; i++){
        for(int j=0; j<BITS[i];j++){
            sizelst.push_back(i);
        }
    }
    int cnt = 0;
    for(multimap<double, int>::iterator it = SortedP.begin(); it != SortedP.end(); it++){
        cnt++;
        huffman_size.insert({it->second, 0});
        huffman_size[it->second] = sizelst[size-cnt];
    }
}

template<typename A, typename B>
std::pair<B,A> flip_pair(const std::pair<A,B> &p){
    return std::pair<B,A>(p.second, p.first);
}
template<typename A, typename B>
std::multimap<B,A> flip_map(const std::map<A,B> &src){
    std::multimap<B,A> dst;
    std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()),
                   flip_pair<A,B>);
    return dst;
}

