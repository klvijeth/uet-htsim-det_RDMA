
# Parameters

The parameters we use in this work are listed in this document.


## Traffic Volume

| date	| volumes_lhcone_in	| volumes_lhcone_out | volumes_oscars_in | volumes_oscars_out | volumes_total_in | volumes_total_out |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- |
|**2025-03** | 91341819211746240 | 91276133597787200 | **26884386150358216** | 24951683434403852 | **187360069836472060** | 185430795856478800 |

![Screenshot 2025-05-03 at 13 02 32](https://github.com/user-attachments/assets/2f321ebb-a1cc-48e2-9cf9-dcfc98cc737f)


## Number of iterations

The number of iterations is set to $277500$, as we have $75$ nodes in the network, and we choose source-destination pairs. 
**The number of pair combinations is $\binom{75}{2} = 2775$**, so we scaled it by $100$ to capture enough varieties of the traffic.


## Packet size

$9000$ bytes, aligning with the size of jumbo frames.
