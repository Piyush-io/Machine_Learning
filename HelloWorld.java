import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class HelloWorld {

    private static class Result {

        int size;
        String searchType;
        String caseType;
        double time;

        Result(int size, String searchType, String caseType, double time) {
            this.size = size;
            this.searchType = searchType;
            this.caseType = caseType;
            this.time = time;
        }

        @Override
        public String toString() {
            return String.format(
                "Size: %d, Search: %s, Case: %s, Time: %.6f ms",
                size,
                searchType,
                caseType,
                time
            );
        }
    }

    private static List<Result> results = new ArrayList<>();

    public static void main(String[] args) {
        int[] sizes = { 5000, 10000, 15000 };
        int iterations = 1000;

        for (int n : sizes) {
            System.out.println("\nArray size: " + n);
            int[] arr = new int[n];
            for (int i = 0; i < n; i++) {
                arr[i] = (int) (Math.random() * 10000);
            }
            Arrays.sort(arr); // Sort once for binary search

            // Best case
            testSearches(arr, n, arr[n / 2], arr[0], "Best", iterations);

            // Worst case
            testSearches(arr, n, -1, -1, "Worst", iterations);

            // Average case
            testSearches(
                arr,
                n,
                arr[(int) (Math.random() * n)],
                arr[(int) (Math.random() * n)],
                "Average",
                iterations
            );
        }

        // Print results
        for (Result result : results) {
            System.out.println(result);
        }
    }

    public static void testSearches(
        int[] arr,
        int n,
        int binaryElement,
        int linearElement,
        String caseType,
        int iterations
    ) {
        System.out.println(caseType + " Case:");

        // BinarySearch
        long totalBinaryTime = 0;
        for (int i = 0; i < iterations; i++) {
            long startBinary = System.nanoTime();
            boolean binaryResult = BinarySearch(arr, n, binaryElement);
            long endBinary = System.nanoTime();
            totalBinaryTime += (endBinary - startBinary);
        }
        double avgBinaryTime = (totalBinaryTime / iterations) / 1_00_000.0;
        results.add(new Result(n, "Binary", caseType, avgBinaryTime));
        System.out.println(
            "BinarySearch average processing time: " + avgBinaryTime + " ns"
        );

        // LinearSearch
        long totalLinearTime = 0;
        for (int i = 0; i < iterations; i++) {
            long startLinear = System.nanoTime();
            boolean linearResult = LinearSearch(arr, n, linearElement);
            long endLinear = System.nanoTime();
            totalLinearTime += (endLinear - startLinear);
        }
        double avgLinearTime = (totalLinearTime / iterations) / 1_00_000.0;
        results.add(new Result(n, "Linear", caseType, avgLinearTime));
        System.out.println(
            "LinearSearch average processing time: " + avgLinearTime + " ns"
        );
    }

    // Binary search method
    public static boolean BinarySearch(int[] arr, int n, int e) {
        int li = 0;
        int ri = n - 1;
        while (li <= ri) {
            int mid = li + (ri - li) / 2;
            if (arr[mid] == e) {
                return true;
            } else if (arr[mid] > e) {
                ri = mid - 1;
            } else {
                li = mid + 1;
            }
        }
        return false;
    }

    // Linear search method
    public static boolean LinearSearch(int[] arr, int n, int e) {
        for (int i = 0; i < n; i++) {
            if (arr[i] == e) {
                return true;
            }
        }
        return false;
    }

    // Method to get results for plotting
    public static List<Result> getResults() {
        return results;
    }
}
