#!/usr/bin/env python3
"""
CLI tool to clean up dangling IPC objects from polymorph_ipc.
"""
import argparse
import sys
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def cleanup_all_pmipc_objects(verbose=False):
    """Clean up all IPC objects with pmipc_ prefix using pattern matching."""
    try:
        import posix_ipc
        import multiprocessing.shared_memory as sm
        import os
    except ImportError as e:
        print(f"Error: Required modules not available: {e}")
        return 0
    
    cleaned_count = 0
    
    if verbose:
        print("Scanning for all pmipc_* semaphores...")
    
    # Clean up semaphores by scanning /dev/shm or using pattern matching
    try:
        # Try to get all semaphore names if available
        if hasattr(posix_ipc, 'sem_list'):
            for sem_name in posix_ipc.sem_list():
                if 'pmipc_' in sem_name:
                    try:
                        posix_ipc.unlink_semaphore(sem_name)
                        if verbose:
                            print(f"  ✓ Cleaned semaphore: {sem_name}")
                        cleaned_count += 1
                    except Exception as e:
                        if verbose:
                            print(f"  ✗ Error cleaning semaphore {sem_name}: {e}")
        else:
            # Fallback: scan /dev/shm directory
            shm_dir = "/dev/shm"
            if os.path.exists(shm_dir):
                for name in os.listdir(shm_dir):
                    if name.startswith("sem.pmipc_"):
                        sem_name = f"/{name[4:]}"  # Remove 'sem.' prefix
                        try:
                            posix_ipc.unlink_semaphore(sem_name)
                            if verbose:
                                print(f"  ✓ Cleaned semaphore: {sem_name}")
                            cleaned_count += 1
                        except Exception as e:
                            if verbose:
                                print(f"  ✗ Error cleaning semaphore {sem_name}: {e}")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Error scanning semaphores: {e}")
    
    # Clean up shared memory objects with pmipc_ prefix
    try:
        shm_dir = "/dev/shm"
        if os.path.exists(shm_dir):
            for name in os.listdir(shm_dir):
                if name.startswith("pmipc_") and ("_metadata" in name or "_pool" in name):
                    try:
                        shm = sm.SharedMemory(name, create=False)
                        shm.close()
                        shm.unlink()
                        if verbose:
                            print(f"  ✓ Cleaned shared memory: {name}")
                        cleaned_count += 1
                    except Exception as e:
                        if verbose:
                            print(f"  ✗ Error cleaning shared memory {name}: {e}")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Error scanning shared memory: {e}")
    
    return cleaned_count

def cleanup_ipc_objects(pool_names=None, verbose=False):
    """Clean up IPC objects for specified pools or all known pools."""
    try:
        import posix_ipc
        import multiprocessing.shared_memory as sm
    except ImportError as e:
        print(f"Error: Required modules not available: {e}")
        return False
    
    # If no specific pool names, try common test pool names
    if pool_names is None:
        pool_names = [
            "test_2d_basic", "test_3d_basic", "test_4d_batch",
            "test_zero_fill_short", "test_zero_fill_long",
            "test_shape_mismatch", "test_dtype_mismatch", 
            "test_type_mismatch", "test_padding_pattern", "test_large_3d",
            # Torch test pools
            "test_torch_cpu", "test_torch_gpu", "test_torch_mismatch_backend",
            "test_torch_mismatch_device", "test_torch_explicit_cpu", "test_torch_explicit_gpu",
            # Frequency test pools
            "test_multiprocess_freq", "test_consumer_first"
        ]
    
    cleaned_count = 0
    
    for pool_name in pool_names:
        if verbose:
            print(f"Cleaning up pool: {pool_name}")
        
        # Clean up semaphores with pmipc prefix
        semaphore_names = [f"/pmipc_{pool_name}_notif", f"/pmipc_{pool_name}_mtx"]
        for sem_name in semaphore_names:
            try:
                posix_ipc.unlink_semaphore(sem_name)
                if verbose:
                    print(f"  ✓ Cleaned semaphore: {sem_name}")
                cleaned_count += 1
            except posix_ipc.ExistentialError:
                if verbose:
                    print(f"  - Semaphore not found: {sem_name}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error cleaning semaphore {sem_name}: {e}")
        
        # Clean up shared memory
        shm_names = [f"{pool_name}_metadata", f"{pool_name}_numpy_pool", f"{pool_name}_torch_pool"]
        for shm_name in shm_names:
            try:
                shm = sm.SharedMemory(shm_name, create=False)
                shm.close()
                shm.unlink()
                if verbose:
                    print(f"  ✓ Cleaned shared memory: {shm_name}")
                cleaned_count += 1
            except FileNotFoundError:
                if verbose:
                    print(f"  - Shared memory not found: {shm_name}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error cleaning shared memory {shm_name}: {e}")
    
    return cleaned_count

def list_ipc_objects(verbose=False):
    """List existing IPC objects."""
    try:
        import posix_ipc
        import multiprocessing.shared_memory as sm
    except ImportError as e:
        print(f"Error: Required modules not available: {e}")
        return
    
    print("Scanning for existing IPC objects...")
    
    # This is a simplified scan - in practice, you might need to check /dev/shm
    # and /proc for a more comprehensive list
    test_pool_names = [
        "test_2d_basic", "test_3d_basic", "test_4d_batch",
        "test_zero_fill_short", "test_zero_fill_long",
        "test_shape_mismatch", "test_dtype_mismatch", 
        "test_type_mismatch", "test_padding_pattern", "test_large_3d",
        # Torch test pools
        "test_torch_cpu", "test_torch_gpu", "test_torch_mismatch_backend",
        "test_torch_mismatch_device", "test_torch_explicit_cpu", "test_torch_explicit_gpu",
        # Frequency test pools
        "test_multiprocess_freq", "test_consumer_first"
    ]
    
    found_objects = []
    
    for pool_name in test_pool_names:
        # Check semaphores with pmipc prefix
        semaphore_names = [f"/pmipc_{pool_name}_notif", f"/pmipc_{pool_name}_mtx"]
        for sem_name in semaphore_names:
            try:
                sem = posix_ipc.Semaphore(sem_name)
                sem.close()
                found_objects.append(f"Semaphore: {sem_name}")
            except posix_ipc.ExistentialError:
                pass
            except Exception:
                pass
        
        # Check shared memory
        shm_names = [f"{pool_name}_metadata", f"{pool_name}_numpy_pool", f"{pool_name}_torch_pool"]
        for shm_name in shm_names:
            try:
                shm = sm.SharedMemory(shm_name)
                shm.close()
                found_objects.append(f"Shared Memory: {shm_name}")
            except FileNotFoundError:
                pass
            except Exception:
                pass
    
    if found_objects:
        print("Found existing IPC objects:")
        for obj in found_objects:
            print(f"  - {obj}")
    else:
        print("No existing IPC objects found.")
    
    return found_objects

def cleanup_registry_files():
    """Clean up registry files."""
    registry_dir = Path("/tmp/tensorpool-fs")
    if not registry_dir.exists():
        print("Registry directory does not exist.")
        return 0
    
    cleaned_count = 0
    for json_file in registry_dir.glob("*.json"):
        try:
            json_file.unlink()
            print(f"  ✓ Cleaned registry file: {json_file.name}")
            cleaned_count += 1
        except Exception as e:
            print(f"  ✗ Error cleaning registry file {json_file.name}: {e}")
    
    return cleaned_count

def main():
    parser = argparse.ArgumentParser(
        description="Clean up dangling IPC objects from polymorph_ipc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s clean                    # Clean up common test pools
  %(prog)s clean --all              # Clean up all known pools
  %(prog)s clean --pmipc            # Clean up all pmipc_* objects (robust cleanup)
  %(prog)s clean --pools pool1 pool2  # Clean up specific pools
  %(prog)s list                     # List existing IPC objects
  %(prog)s clean --registry         # Clean up registry files only
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean up IPC objects')
    clean_parser.add_argument('--pools', nargs='*', help='Specific pool names to clean')
    clean_parser.add_argument('--all', action='store_true', help='Clean all known pools')
    clean_parser.add_argument('--pmipc', action='store_true', help='Clean all pmipc_* objects using pattern matching')
    clean_parser.add_argument('--registry', action='store_true', help='Clean registry files only')
    clean_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List existing IPC objects')
    list_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.command == 'clean':
        print("🧹 Cleaning up IPC objects...")
        
        cleaned_ipc = 0
        cleaned_registry = 0
        
        if args.registry:
            # Only clean registry files
            cleaned_registry = cleanup_registry_files()
        elif args.pmipc:
            # Clean all pmipc_* objects using pattern matching
            cleaned_ipc = cleanup_all_pmipc_objects(args.verbose)
        else:
            # Clean IPC objects
            pool_names = None
            if args.pools:
                pool_names = args.pools
            elif not args.all:
                # Default to common test pools
                pool_names = None
            
            cleaned_ipc = cleanup_ipc_objects(pool_names, args.verbose)
            
            # Also clean registry files
            if args.verbose:
                print("\nCleaning registry files...")
            cleaned_registry = cleanup_registry_files()
        
        print("\n✅ Cleanup completed!")
        print(f"   - IPC objects cleaned: {cleaned_ipc}")
        print(f"   - Registry files cleaned: {cleaned_registry}")
        
    elif args.command == 'list':
        found_objects = list_ipc_objects(args.verbose)
        
        if found_objects is not None:
            print(f"\n📊 Found {len(found_objects)} IPC objects")
        else:
            print("\n📊 Could not scan for IPC objects")
        
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
