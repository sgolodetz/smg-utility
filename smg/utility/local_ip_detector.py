from netifaces import AF_INET, ifaddresses, interfaces
from typing import List, Optional


class LocalIPDetector:
    """TODO"""

    # PUBLIC STATIC METHODS

    @staticmethod
    def get_all_ips() -> List[str]:
        """
        TODO

        :return:    TODO
        """
        result = []  # type: List[str]
        for interface in interfaces():
            for link in ifaddresses(interface).get(AF_INET, []):
                result.append(link["addr"])
        return result


    @staticmethod
    def get_ip_starting_with(partial_ip: str) -> Optional[str]:
        """
        TODO

        :param partial_ip:  TODO
        :return:            TODO
        """
        for ip in LocalIPDetector.get_all_ips():
            if ip.startswith(partial_ip):
                return ip
        return None


def main() -> None:
    print(LocalIPDetector.get_all_ips())
    print(LocalIPDetector.get_ip_starting_with("163.1.88"))

if __name__ == "__main__":
    main()
